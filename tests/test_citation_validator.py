from __future__ import annotations

from email_thread_rag.rag.answer import DraftAnswer, DraftClause
from email_thread_rag.rag.citation_validator import CitationValidator


def test_citation_validator_drops_unsupported_claims(sample_records):
    _, _, _, chunks = sample_records
    validator = CitationValidator()
    hit = next(chunk for chunk in chunks if chunk.chunk_id.endswith("msg-2-att-1-page-1-chunk-0"))
    from email_thread_rag.app.schemas import RetrievalHit

    draft = DraftAnswer(
        clauses=[
            DraftClause(
                text="The final budget amount is $1500.",
                supporting_hits=[RetrievalHit(chunk=hit)],
            ),
            DraftClause(
                text="Bob Director approved the final budget.",
                supporting_hits=[RetrievalHit(chunk=hit)],
            ),
            DraftClause(
                text="Acme Supplies appears in the final budget.",
                supporting_hits=[RetrievalHit(chunk=hit)],
            ),
            DraftClause(
                text="The vendor is Contoso.",
                supporting_hits=[RetrievalHit(chunk=hit)],
            ),
        ]
    )
    result = validator.validate(draft, draft.clauses[0].supporting_hits)
    assert "Contoso" not in result.answer
    assert any(not clause.kept for clause in result.clause_validations)
    assert result.answer == "I could not support enough of that answer from the selected evidence." or "$1500" in result.answer
