from supplychain_copilot.rag.pipeline import get_retriever


def test_retrieves_payment_terms():
    hits = get_retriever().search("what are Acme payment terms", k=3)
    assert hits
    top_text = " ".join(chunk.text.lower() for chunk, _ in hits)
    assert "net 45" in top_text or "payment" in top_text


def test_retrieves_from_correct_contract():
    hits = get_retriever().search("Shenzhen late delivery penalty", k=3)
    sources = {chunk.source for chunk, _ in hits}
    assert "supplier_contract_shenzhen.md" in sources


def test_scores_ordered_descending():
    hits = get_retriever().search("safety stock policy", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
