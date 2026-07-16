from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.rag.utils import normalize_subject


def reconstruct_threads(emails: list[EmailRecord]) -> list[EmailRecord]:
    by_message_id = {email.message_id: email for email in emails}
    thread_lookup: dict[str, str] = {}

    for email in sorted(emails, key=lambda item: item.date):
        if email.in_reply_to and email.in_reply_to in by_message_id:
            thread_lookup[email.message_id] = by_message_id[email.in_reply_to].thread_id
            continue
        reference_match = next((ref for ref in email.references if ref in by_message_id), None)
        if reference_match:
            thread_lookup[email.message_id] = by_message_id[reference_match].thread_id
            continue
        thread_lookup[email.message_id] = email.thread_id

    grouped_by_subject: dict[str, list[EmailRecord]] = defaultdict(list)
    for email in emails:
        grouped_by_subject[normalize_subject(email.subject)].append(email)

    for subject_group in grouped_by_subject.values():
        ordered = sorted(subject_group, key=lambda item: item.date)
        for current in ordered:
            if current.message_id in thread_lookup and thread_lookup[current.message_id] != current.thread_id:
                continue
            for candidate in reversed(ordered):
                if candidate.message_id == current.message_id:
                    continue
                participant_overlap = set(current.to + current.cc + [current.sender]) & set(
                    candidate.to + candidate.cc + [candidate.sender]
                )
                if participant_overlap and abs(current.date - candidate.date) <= timedelta(days=7):
                    thread_lookup[current.message_id] = thread_lookup.get(candidate.message_id, candidate.thread_id)
                    break

    updated: list[EmailRecord] = []
    for email in emails:
        updated.append(email.model_copy(update={"thread_id": thread_lookup.get(email.message_id, email.thread_id)}))
    return updated

