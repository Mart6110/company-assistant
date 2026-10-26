from pathlib import Path

import pytest

from assistant.guardrails.classifier import Classification
from assistant.llm import ModelRef
from assistant.rag.chunker import SourceDocument, chunk_document, load_document
from assistant.rag.embeddings import HashingEmbedder
from assistant.router import ModelRouter
from tests.fakes import ROOT


def test_load_document_front_matter():
    doc = load_document(ROOT / "data/policies/compensation-bands.md")
    assert doc.doc_id == "hr-comp-001"
    assert doc.allowed_roles == ["hr", "admin"]
    assert doc.classification == "confidential"
    assert not doc.text.startswith("---")


def test_chunks_carry_heading_breadcrumbs():
    doc = load_document(ROOT / "data/policies/annual-leave.md")
    chunks = chunk_document(doc)
    headings = {c.heading for c in chunks}
    assert "Carry-over" in headings and "Sick leave" in headings
    carry = next(c for c in chunks if c.heading == "Carry-over")
    assert "31 March" in carry.text
    assert carry.embedding_text(doc.title).startswith("Annual Leave Policy › Carry-over")


def test_long_sections_are_split_with_overlap():
    para = "This sentence is about leave rules and approvals. " * 10
    doc = SourceDocument("d", "T", "# T\n\n## S\n\n" + "\n\n".join([para.strip()] * 6), "x", ["employee"])
    chunks = chunk_document(doc, max_chars=1200, overlap_chars=600)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1200 * 2 for c in chunks)
    assert chunks[1].text.startswith(chunks[0].text.split("\n\n")[-1])


def test_missing_front_matter_fields(tmp_path: Path):
    p = tmp_path / "bad.md"
    p.write_text("---\ntitle: No id\n---\nbody")
    with pytest.raises(ValueError, match="doc_id"):
        load_document(p)


async def test_hashing_embedder_is_normalized_and_deterministic():
    e = HashingEmbedder(128)
    [a, b] = await e.embed(["annual leave carry over", "annual leave carry over"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-5


ROUTER = ModelRouter(ModelRef("anthropic", "claude-haiku-5-5"), ModelRef("anthropic", "claude-opus-5-5"))


@pytest.mark.parametrize(
    ("query", "classification", "tier"),
    [
        ("How many vacation days do I get?", None, "small"),
        ("What's the status of IT-1042?", Classification("safe", "account_info", "simple", ""), "small"),
        (
            "If I start on 1 April and work 80%, how many days do I get, and can I carry them over?",
            Classification("safe", "policy_question", "complex", ""),
            "frontier",
        ),
        ("Draft a message to my team explaining the new hybrid rules and compare them to the old ones", None, "small"),
        (
            "Draft a message to my team explaining the new hybrid rules and compare them to the old ones",
            Classification("safe", "policy_question", "complex", ""),
            "frontier",
        ),
    ],
)
def test_router(query, classification, tier):
    assert ROUTER.route(query, classification=classification).tier == tier


def test_router_force_and_alternate():
    d = ROUTER.route("hi", force_tier="frontier")
    assert d.tier == "frontier" and d.model.model == "claude-opus-5-5"
    assert ROUTER.alternate(d).model == "claude-haiku-5-5"


def test_model_ref_parse():
    assert ModelRef.parse("azure:my-gpt-deployment", "low") == ModelRef("azure", "my-gpt-deployment", "low")
    with pytest.raises(ValueError):
        ModelRef.parse("claude-opus-5-5")
