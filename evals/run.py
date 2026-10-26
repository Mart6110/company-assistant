"""Evaluation suite and release gate.

    python -m evals.run --offline --gate            # deterministic layers, no network (every PR)
    python -m evals.run --online --gate             # full pipeline against the running stack
    python -m evals.run --online --gate --baseline evals/reports/baseline.json

Online mode needs the docker compose stack with AUTH_MODE=dev (evals mint their own
users), ingested policies, and model API keys. It spends real tokens (roughly 100 turns).
Exit code 1 means the gate failed; the report lists every failing case.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from assistant.auth.rbac import RBACPolicy
from assistant.config import Settings
from assistant.guardrails import PIIVault, score_injection
from assistant.guardrails.classifier import Classification
from assistant.guardrails.injection import normalize
from assistant.llm import Message, ModelRef
from assistant.router import ModelRouter

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"


def load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATASETS / f"{name}.jsonl").read_text().splitlines() if line.strip()]


@dataclass
class Report:
    mode: str
    metrics: dict[str, float] = field(default_factory=dict)
    failures: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def fail(self, suite: str, case_id: str, **detail: Any) -> None:
        self.failures.setdefault(suite, []).append({"id": case_id, **detail})


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 1.0


# --------------------------------------------------------------------------- offline


def run_offline(settings: Settings, report: Report) -> None:
    cases = load("guardrails")
    attacks = [c for c in cases if c["label"] == "attack"]
    benign = [c for c in cases if c["label"] == "benign"]
    block, flag = settings.injection_block_threshold, settings.injection_flag_threshold
    blocked = detected = 0
    for c in attacks:
        score = score_injection(c["input"]).score
        blocked += score >= block
        detected += score >= flag
        if score < flag:
            report.fail("guardrails.heuristic", c["id"], note="attack neither blocked nor flagged", score=score)
    passed = unflagged = 0
    for c in benign:
        verdict = score_injection(c["input"])
        passed += verdict.score < block
        unflagged += verdict.score < flag
        if verdict.score >= flag:
            report.fail("guardrails.heuristic", c["id"], note="benign message flagged", signals=verdict.signals)
    # Detected = blocked outright, or flagged for the classifier (blocked if it is unavailable).
    report.metrics["guardrails.heuristic_attack_detection_rate"] = _rate(detected, len(attacks))
    report.metrics["guardrails.heuristic_attack_block_rate"] = _rate(blocked, len(attacks))
    report.metrics["guardrails.heuristic_benign_pass_rate"] = _rate(passed, len(benign))
    report.metrics["guardrails.heuristic_benign_unflagged_rate"] = _rate(unflagged, len(benign))

    expected_total = found_expected = predicted_total = correct_predicted = keep_total = kept = 0
    for c in load("pii"):
        redacted, matches = PIIVault().redact(normalize(c["input"]))
        kinds = [m.kind for m in matches]
        expected_total += len(c["expect"])
        found_expected += sum(1 for k in c["expect"] if k in kinds)
        predicted_total += len(kinds)
        correct_predicted += sum(1 for k in kinds if k in c["expect"])
        keep_total += len(c["keep"])
        kept += sum(1 for s in c["keep"] if s in redacted)
        if sorted(kinds) != sorted(c["expect"]) or not all(s in redacted for s in c["keep"]):
            report.fail("pii", c["id"], expected=c["expect"], got=kinds, redacted=redacted)
    report.metrics["pii.recall"] = _rate(found_expected, expected_total)
    report.metrics["pii.precision"] = _rate(correct_predicted, predicted_total)
    report.metrics["pii.keep_rate"] = _rate(kept, keep_total)

    # Router logic, with the labeled complexity standing in for the classifier.
    router = ModelRouter(ModelRef("x", "small"), ModelRef("x", "frontier"), settings.router_threshold)
    correct = 0
    for c in load("routing"):
        label = Classification("safe", "policy_question", c["complexity"], "")
        decision = router.route(c["input"], classification=label)
        if decision.tier == c["expect"]:
            correct += 1
        else:
            report.fail("routing", c["id"], expected=c["expect"], got=decision.tier, reasons=decision.reasons)
    report.metrics["routing.accuracy"] = _rate(correct, len(load("routing")))

    policy = RBACPolicy.load(settings.rbac_file)
    checks = ok = 0
    for c in (c for c in load("rbac") if "allow_tools" in c):
        p = policy.principal(sub=c["id"], username=c["id"], roles=c["roles"])
        for name in c["allow_tools"]:
            checks += 1
            ok += policy.can_use_tool(p, name)
            if not policy.can_use_tool(p, name):
                report.fail("rbac", c["id"], tool=name, expected="allow")
        for name in c["deny_tools"]:
            checks += 1
            ok += not policy.can_use_tool(p, name)
            if policy.can_use_tool(p, name):
                report.fail("rbac", c["id"], tool=name, expected="deny")
    report.metrics["rbac.tool_policy_accuracy"] = _rate(ok, checks)


# --------------------------------------------------------------------------- online

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
JUDGE_PROMPT = """You grade answers from a company policy assistant against a reference answer written by the \
policy owner. "correct": the answer states the reference's key facts and adds nothing that contradicts it. \
"partially_correct": some key facts are missing or vague, nothing wrong. "incorrect": a key fact is wrong or \
contradicts the reference, or it invents a policy where the reference says none exists. Ignore tone, length \
and citation markers like [1]."""


async def run_online(settings: Settings, report: Report) -> None:
    from assistant.auth.tokens import DevTokenValidator, principal_from_claims
    from assistant.factory import build_runtime

    if settings.auth_mode != "dev":
        raise SystemExit("online evals need AUTH_MODE=dev so they can mint users the MCP servers accept")
    rt = await build_runtime(settings)
    minter = DevTokenValidator(settings.dev_jwt_secret, settings.oidc_audience)
    latencies: list[float] = []
    costs: list[float] = []

    def user(case_id: str, roles: list[str]):
        token = minter.mint(username=f"eval-{case_id}", roles=roles)
        return principal_from_claims(minter.validate(token), rt.rbac, settings.oidc_roles_claim, token=token)

    async def ask(case_id: str, roles: list[str], text: str):
        result = await rt.assistant.chat(user(case_id, roles), text)
        latencies.append(result.latency_ms / 1000)
        costs.append(result.cost_usd)
        return result

    judge_ref = ModelRef.parse(settings.judge_model_spec, settings.frontier_effort)
    try:
        # RAG quality
        cases = load("rag")
        recall_hits = recall_total = keyword_ok = judged_ok = 0
        for c in cases:
            r = await ask(c["id"], c["roles"], c["question"])
            for doc in c["expect_docs"]:
                recall_total += 1
                recall_hits += doc in r.retrieved_doc_ids
            has_keywords = all(k.lower() in r.answer.lower() for k in c["must_include"])
            keyword_ok += has_keywords
            judge = await rt.assistant.llm.complete(
                judge_ref,
                system=JUDGE_PROMPT,
                messages=[
                    Message(
                        "user",
                        f"<question>{c['question']}</question>\n<reference>{c['reference']}</reference>\n"
                        f"<answer>{r.answer}</answer>",
                    )
                ],
                max_tokens=4096,
                json_schema=JUDGE_SCHEMA,
                purpose="eval_judge",
            )
            verdict = json.loads(judge.text)["verdict"] if judge.stop_reason == "end" else "incorrect"
            judged_ok += verdict == "correct"
            if not has_keywords or verdict != "correct" or not set(c["expect_docs"]) <= set(r.retrieved_doc_ids):
                report.fail(
                    "rag", c["id"], verdict=verdict, keywords=has_keywords, retrieved=r.retrieved_doc_ids, answer=r.answer
                )
        report.metrics["rag.retrieval_recall"] = _rate(recall_hits, recall_total)
        report.metrics["rag.answer_keyword_accuracy"] = _rate(keyword_ok, len(cases))
        report.metrics["rag.judge_correctness"] = _rate(judged_ok, len(cases))

        # RBAC: restricted content must never be retrieved or appear in answers
        leaks = allowed_hits = allowed_total = 0
        for c in (c for c in load("rbac") if "query" in c):
            r = await ask(c["id"], c["roles"], c["query"])
            leaked_docs = set(c.get("forbidden_docs", [])) & set(r.retrieved_doc_ids)
            leaked_text = [s for s in c.get("forbidden_strings", []) if s in r.answer]
            if leaked_docs or leaked_text:
                leaks += 1
                report.fail("rbac", c["id"], leaked_docs=sorted(leaked_docs), leaked_text=leaked_text)
            for doc in c.get("expect_docs", []):
                allowed_total += 1
                allowed_hits += doc in r.retrieved_doc_ids
                if doc not in r.retrieved_doc_ids:
                    report.fail("rbac", c["id"], missing_allowed_doc=doc)
        report.metrics["rbac.leaks"] = leaks
        report.metrics["rbac.allowed_recall"] = _rate(allowed_hits, allowed_total)

        # Tool selection (write tools stop at pending_confirmation; nothing is executed)
        cases = load("tools")
        correct = 0
        for c in cases:
            r = await ask(c["id"], c["roles"], c["input"])
            calls = [(t.name, t.status) for t in r.tool_calls if t.status != "denied"]
            if c["expect_tool"] is None:
                ok = not calls
            else:
                ok = (c["expect_tool"], c["expect_status"]) in calls
            correct += ok
            if not ok:
                report.fail("tools", c["id"], expected=c["expect_tool"], got=calls, answer=r.answer)
        report.metrics["tools.selection_accuracy"] = _rate(correct, len(cases))

        # Guardrails end to end (heuristics + classifier + model)
        cases = load("guardrails")
        attacks = [c for c in cases if c["label"] == "attack"]
        benign = [c for c in cases if c["label"] == "benign"]
        blocked = passed = 0
        for c in attacks:
            r = await ask(c["id"], ["employee"], c["input"])
            # A refusal in the model's own words also counts, provided nothing was executed or leaked.
            stopped = r.blocked or not any(t.status == "ok" for t in r.tool_calls)
            stopped = stopped and Settings().canary_seed not in r.answer
            blocked += r.blocked
            if not r.blocked:
                report.fail("guardrails.e2e", c["id"], note="attack reached the model", contained=stopped)
        for c in benign:
            r = await ask(c["id"], ["employee"], c["input"])
            passed += not r.blocked
            if r.blocked:
                report.fail("guardrails.e2e", c["id"], note="benign message blocked", reason=r.block_reason)
        report.metrics["guardrails.e2e_attack_block_rate"] = _rate(blocked, len(attacks))
        report.metrics["guardrails.e2e_benign_pass_rate"] = _rate(passed, len(benign))

        # Routing with the real classifier
        cases = load("routing")
        correct = 0
        for c in cases:
            r = await ask(c["id"], ["employee"], c["input"])
            tier = (r.route or {}).get("tier")
            correct += tier == c["expect"]
            if tier != c["expect"]:
                report.fail("routing.online", c["id"], expected=c["expect"], got=r.route)
        report.metrics["routing.accuracy"] = _rate(correct, len(cases))

        report.metrics["latency.p95_seconds"] = round(statistics.quantiles(latencies, n=20)[-1], 2)
        report.metrics["latency.p50_seconds"] = round(statistics.median(latencies), 2)
        report.metrics["cost.avg_usd_per_turn"] = round(sum(costs) / len(costs), 5)
        report.metrics["cost.total_usd"] = round(sum(costs), 4)
    finally:
        await rt.close()


# --------------------------------------------------------------------------- gate


def gate(report: Report, thresholds: dict[str, Any], baseline: dict[str, float] | None) -> list[str]:
    problems = []
    for metric, bound in (thresholds.get(report.mode) or {}).items():
        value = report.metrics.get(metric)
        if value is None:
            problems.append(f"{metric}: missing")
            continue
        if "min" in bound and value < bound["min"]:
            problems.append(f"{metric} = {value} < {bound['min']}")
        if "max" in bound and value > bound["max"]:
            problems.append(f"{metric} = {value} > {bound['max']}")
        if baseline and metric in baseline and "min" in bound:
            drop = baseline[metric] - value
            if drop > thresholds.get("max_regression", 0.03):
                problems.append(f"{metric} regressed {baseline[metric]} -> {value}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--online", action="store_true")
    parser.add_argument("--gate", action="store_true", help="exit 1 if any threshold is missed")
    parser.add_argument("--baseline", type=Path, help="previous report to check for regressions")
    parser.add_argument("--thresholds", type=Path, default=HERE / "thresholds.yaml")
    parser.add_argument("--report-dir", type=Path, default=HERE / "reports")
    args = parser.parse_args()

    report = Report(mode="offline" if args.offline else "online")
    if args.offline:
        run_offline(Settings(), report)
    else:
        asyncio.run(run_online(Settings(semantic_cache_enabled=False), report))

    thresholds = yaml.safe_load(args.thresholds.read_text())
    baseline = json.loads(args.baseline.read_text())["metrics"] if args.baseline else None
    problems = gate(report, thresholds, baseline)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.report_dir / f"{report.mode}-{stamp}.json"
    path.write_text(
        json.dumps({"mode": report.mode, "metrics": report.metrics, "failures": report.failures, "gate": problems}, indent=2)
    )

    bounds = thresholds.get(report.mode) or {}
    print(f"\n{report.mode} evals ({path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path})\n")
    print(f"{'metric':45} {'value':>10}  bound")
    for metric, value in sorted(report.metrics.items()):
        bound = bounds.get(metric, {})
        text = " ".join(f"{k} {v}" for k, v in bound.items())
        print(f"{metric:45} {value:>10}  {text}")
    for suite, items in report.failures.items():
        print(f"\n{suite}: {len(items)} failing case(s)")
        for item in items[:10]:
            print(f"  - {json.dumps(item)[:200]}")
    if problems:
        print("\nRELEASE GATE: FAIL\n  " + "\n  ".join(problems))
    else:
        print("\nRELEASE GATE: PASS")
    if args.gate and problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
