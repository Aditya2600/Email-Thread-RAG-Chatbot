from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from email_thread_rag.rag.utils import normalize_subject, read_json


MESSAGE_KEYS = {"message_id", "Message-ID", "id", "subject", "body", "plain_text_body", "text"}


def _looks_like_message(node: dict[str, Any]) -> bool:
    return len(MESSAGE_KEYS.intersection(node.keys())) >= 2


def _walk_messages(node: Any, sink: list[dict[str, Any]]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_messages(item, sink)
        return

    if not isinstance(node, dict):
        return

    if _looks_like_message(node):
        sink.append(node)
        return

    for value in node.values():
        if isinstance(value, (dict, list)):
            _walk_messages(value, sink)


def load_mailbox_messages(paths: list[Path]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        _walk_messages(payload, messages)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in messages:
        message_id = str(message.get("message_id") or message.get("Message-ID") or message.get("id") or "").strip()
        if not message_id or message_id in seen:
            continue
        seen.add(message_id)
        unique.append(message)
    return unique


def _parse_datetime(value: Any) -> datetime:
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    parsed = date_parser.parse(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_party(value: Any) -> str:
    if isinstance(value, dict):
        email = str(value.get("email") or "").strip()
        name = str(value.get("name") or "").strip()
        return email or name
    return str(value or "").strip()


def _coerce_party_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_coerce_party(entry) for entry in value) if item]
    single = _coerce_party(value)
    return [single] if single else []


def _html_to_text(value: str) -> str:
    if "<" not in value or ">" not in value:
        return value.strip()
    soup = BeautifulSoup(value, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(chunk.strip() for chunk in soup.stripped_strings if chunk.strip())
    return text.strip()


def _attachment_source_path(attachment: dict[str, Any]) -> str | None:
    for key in ("repo_path", "path", "relative_path"):
        value = attachment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    url = attachment.get("url")
    if isinstance(url, str) and url.strip():
        parsed = urlparse(url)
        return parsed.path.lstrip("/")
    return None


def _attachment_filename(attachment: dict[str, Any], fallback_index: int) -> str:
    for key in ("filename", "name", "file_name", "basename"):
        value = attachment.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).name
    source_path = _attachment_source_path(attachment)
    if source_path:
        return Path(source_path).name
    return f"attachment-{fallback_index}"


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip("<>"))
    return cleaned or "message"


def _has_attachments(message: dict[str, Any]) -> bool:
    return bool(message.get("attachments"))


def _thread_attachment_score(messages: list[dict[str, Any]]) -> tuple[int, int]:
    attachment_message_count = sum(1 for message in messages if _has_attachments(message))
    attachment_count = sum(len(message.get("attachments", [])) for message in messages)
    return attachment_message_count, attachment_count


def _filter_messages_by_date(messages: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    start_value = selection.get("date_start")
    end_value = selection.get("date_end")
    if not start_value and not end_value:
        return messages

    start = _parse_datetime(start_value) if start_value else datetime.min.replace(tzinfo=timezone.utc)
    end = _parse_datetime(end_value) if end_value else datetime.max.replace(tzinfo=timezone.utc)
    return [message for message in messages if start <= _parse_datetime(message.get("date")) <= end]


def _select_attachment_threads(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    attachment_thread_count: int,
    total_thread_target: int,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    attachment_threads: list[list[dict[str, Any]]] = []
    non_attachment_threads: list[list[dict[str, Any]]] = []
    for messages in grouped.values():
        if any(_has_attachments(message) for message in messages):
            attachment_threads.append(messages)
        else:
            non_attachment_threads.append(messages)

    attachment_threads.sort(
        key=lambda messages: (
            min(len(message.get("attachments", [])) for message in messages if _has_attachments(message)),
            sum(len(message.get("attachments", [])) for message in messages),
            -len(messages),
            messages[0]["thread_id"],
        )
    )
    non_attachment_threads.sort(key=lambda messages: (-len(messages), messages[0]["thread_id"]))

    chosen_attachment_threads = attachment_threads[:attachment_thread_count]
    remaining_threads_needed = max(0, total_thread_target - len(chosen_attachment_threads))
    chosen_non_attachment_threads = non_attachment_threads[:remaining_threads_needed]

    if len(chosen_non_attachment_threads) < remaining_threads_needed:
        extra_attachment_threads = attachment_threads[len(chosen_attachment_threads) : len(chosen_attachment_threads) + (remaining_threads_needed - len(chosen_non_attachment_threads))]
        chosen_attachment_threads.extend(extra_attachment_threads)

    return chosen_attachment_threads, chosen_non_attachment_threads


def normalize_mailbox_message(message: dict[str, Any], mailbox: str) -> dict[str, Any]:
    message_id = str(message.get("message_id") or message.get("Message-ID") or message.get("id") or "").strip()
    subject = str(message.get("subject") or "").strip()
    raw_body = str(
        message.get("body")
        or message.get("plain_text_body")
        or message.get("body_text")
        or message.get("text")
        or message.get("content")
        or ""
    ).strip()
    body = _html_to_text(raw_body)
    thread_key = str(
        message.get("thread_id")
        or message.get("conversation_id")
        or message.get("thread")
        or normalize_subject(subject)
        or message_id
    )
    attachments: list[dict[str, Any]] = []
    for index, attachment in enumerate(message.get("attachments") or message.get("files") or [], start=1):
        if not isinstance(attachment, dict):
            continue
        filename = _attachment_filename(attachment, index)
        normalized = {
            "attachment_id": str(attachment.get("attachment_id") or f"{_sanitize(message_id)}-att-{index}"),
            "filename": filename,
        }
        if attachment.get("local_source"):
            normalized["local_source"] = str(attachment["local_source"])
        if attachment.get("url"):
            normalized["url"] = str(attachment["url"])
        source_path = _attachment_source_path(attachment)
        if source_path:
            normalized["repo_path"] = source_path
        attachments.append(normalized)

    return {
        "doc_id": str(message.get("doc_id") or _sanitize(message_id)),
        "message_id": message_id,
        "thread_id": f"{mailbox}:{thread_key}",
        "date": _parse_datetime(message.get("date")).isoformat(),
        "from": _coerce_party(message.get("from") or message.get("sender") or ""),
        "to": _coerce_party_list(message.get("to")),
        "cc": _coerce_party_list(message.get("cc")),
        "subject": subject,
        "body": body,
        "in_reply_to": message.get("in_reply_to") or message.get("In-Reply-To"),
        "references": _coerce_party_list(message.get("references") or message.get("References") or []),
        "attachments": attachments,
    }


def select_mailbox_messages(mailbox: str, messages: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = [normalize_mailbox_message(message, mailbox) for message in messages]
    normalized = _filter_messages_by_date(normalized, selection)
    selected_ids = {str(value) for value in selection.get("message_ids", [])}
    selected_threads = {f"{mailbox}:{value}" for value in selection.get("thread_ids", [])}

    if selected_ids:
        return [message for message in normalized if message["message_id"] in selected_ids]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in normalized:
        grouped[message["thread_id"]].append(message)

    max_threads = int(selection.get("max_threads", 20))
    max_messages_per_thread = int(selection.get("max_messages_per_thread", 10))
    prefer_attachments = selection.get("prefer_attachments", True)
    max_attachment_messages_per_thread = int(selection.get("max_attachment_messages_per_thread", 1))
    attachment_thread_count = int(selection.get("attachment_thread_count", min(max_threads, 12)))

    if selected_threads:
        ordered_threads = [
            messages
            for thread_id, messages in grouped.items()
            if thread_id in selected_threads
        ]
        ordered_threads.sort(key=lambda messages: messages[0]["thread_id"])
    elif prefer_attachments:
        attachment_threads, non_attachment_threads = _select_attachment_threads(
            grouped,
            attachment_thread_count=attachment_thread_count,
            total_thread_target=max_threads,
        )
        ordered_threads = attachment_threads + non_attachment_threads
    else:
        ordered_threads = [
            messages
            for _, messages in sorted(grouped.items(), key=lambda item: item[0])
        ]

    chosen: list[dict[str, Any]] = []
    for thread_messages in ordered_threads[:max_threads]:
        ordered_by_time = sorted(
            thread_messages,
            key=lambda item: (_parse_datetime(item.get("date")), item.get("message_id", "")),
        )
        if prefer_attachments and any(_has_attachments(message) for message in ordered_by_time):
            attachment_messages = [message for message in ordered_by_time if _has_attachments(message)]
            non_attachment_messages = [message for message in ordered_by_time if not _has_attachments(message)]
            selected_messages = attachment_messages[:max_attachment_messages_per_thread]
            selected_ids = {message["message_id"] for message in selected_messages}
            fill_candidates = non_attachment_messages + [
                message for message in attachment_messages[max_attachment_messages_per_thread:] if message["message_id"] not in selected_ids
            ]
            selected_messages.extend(fill_candidates[: max(0, max_messages_per_thread - len(selected_messages))])
        else:
            selected_messages = ordered_by_time[:max_messages_per_thread]
        chosen.extend(
            sorted(
                selected_messages,
                key=lambda item: (_parse_datetime(item.get("date")), item.get("message_id", "")),
            )
        )
    return chosen


def write_selected_message(message: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(message, handle, indent=2, ensure_ascii=True)
