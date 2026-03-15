from __future__ import annotations

from datetime import datetime, timezone

from email_thread_rag.app.schemas import ChunkRecord, RetrievalHit, RetrievalMetrics
from email_thread_rag.rag.answer import AnswerBuilder


def _hit(
    *,
    chunk_id: str,
    thread_id: str,
    message_id: str,
    subject: str,
    text: str,
    support: float = 1.0,
) -> RetrievalHit:
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        doc_id=message_id,
        thread_id=thread_id,
        message_id=message_id,
        kind="email",
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        sender="sender@example.com",
        subject=subject,
        text=text,
        token_count=len(text.split()),
        source_path="/tmp/test.json",
        source_type="fixture",
    )
    metrics = RetrievalMetrics(chunk_support_score=support)
    return RetrievalHit(chunk=chunk, metrics=metrics)


def test_topic_question_prefers_topic_email_over_forwarded_reply():
    builder = AnswerBuilder()
    hits = [
        _hit(
            chunk_id="forwarded",
            thread_id="thread-presentation",
            message_id="msg-forwarded",
            subject="RE: Presentation to Trading Track A&A",
            text="Hi Philip If you do have slides prepared, can you have your assistant e:mail a copy to Mog Heu, who will conference in from New York.",
            support=1.0,
        ),
        _hit(
            chunk_id="topic",
            thread_id="thread-presentation",
            message_id="msg-topic",
            subject="Re: Presentation to Trading Track A&A",
            text="The topic will the the western natural gas market. I may have overhead slides. I will bring handouts.",
            support=0.8,
        ),
    ]

    draft = builder.build_direct("What topic was the presentation about?", hits)

    assert draft.clauses
    assert draft.clauses[0].text == "Topic: western natural gas market"
    assert draft.clauses[0].supporting_hits[0].chunk.message_id == "msg-topic"


def test_instructions_question_prefers_instruction_email_over_acknowledgement():
    builder = AnswerBuilder()
    hits = [
        _hit(
            chunk_id="ack",
            thread_id="thread-ferc",
            message_id="msg-ack",
            subject="Re: Instructions for FERC Meetings",
            text="it works. thank you",
            support=1.0,
        ),
        _hit(
            chunk_id="instructions",
            thread_id="thread-ferc",
            message_id="msg-instructions",
            subject="Instructions for FERC Meetings",
            text="Mr. Allen - Per our phone conversation, please see the instructions below to get access to view FERC meetings. Please advise if there are any problems, questions or concerns.",
            support=0.9,
        ),
    ]

    draft = builder.build_direct("What were the instructions about?", hits)

    assert draft.clauses
    assert draft.clauses[0].text == "Instructions: get access to view FERC meetings"
    assert draft.clauses[0].supporting_hits[0].chunk.message_id == "msg-instructions"


def test_temporal_ferc_question_does_not_get_hijacked_by_instruction_intent():
    builder = AnswerBuilder()
    hits = [
        _hit(
            chunk_id="instructions",
            thread_id="thread-ferc",
            message_id="msg-instructions",
            subject="Instructions for FERC Meetings",
            text=(
                "Mr. Allen - Per our phone conversation, please see the instructions below to get access to "
                "view FERC meetings. As long as you are configured to receive Real Video, you should be able "
                "to access the FERC meeting this Wednesday, November 8."
            ),
            support=0.9,
        ),
    ]

    draft = builder.build_direct("instructions when was the FERC meeting mentioned in that message?", hits)

    assert draft.clauses
    assert draft.clauses[0].text == "FERC meeting mentioned for: this Wednesday, November 8"
    assert draft.clauses[0].supporting_hits[0].chunk.message_id == "msg-instructions"


def test_company_question_extracts_attachment_requesting_company():
    builder = AnswerBuilder()
    hits = [
        _hit(
            chunk_id="attachment-company",
            thread_id="thread-bids",
            message_id="msg-company",
            subject="Bid Solicitation",
            text="Southwest Gas Corporation April 2001 Supply Requests",
            support=0.9,
        ),
    ]

    draft = builder.build_direct("What company is requesting bids in the attachment?", hits)

    assert draft.clauses
    assert draft.clauses[0].text == "Company: Southwest Gas Corporation"
    assert draft.clauses[0].supporting_hits[0].chunk.message_id == "msg-company"
