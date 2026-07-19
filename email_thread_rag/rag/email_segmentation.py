"""Deterministic, offline email body segmentation and header construction.

Stage 1: split a raw email body into the sender's newly authored content vs.
quoted reply history, signature, and legal/boilerplate disclaimer. Only the
authored text is chunked and indexed; the rest is retained for audit.

All rules here are conservative and deterministic. No network, no model, no
LLM. Known limitations are documented at the bottom of this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EmailSegments:
    """Result of splitting an email body. Offsets are into ``normalized``."""

    authored_text: str
    quoted_text: str
    signature_text: str
    disclaimer_text: str
    normalized: str
    # [start, end) offset of ``authored_text`` inside ``normalized``.
    authored_start: int
    authored_end: int


# --- Quote markers: everything from the first match onward is quoted history.
_QUOTE_MARKERS = [
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Forwarded (?:by|message)", re.IGNORECASE),
    # "On <date>, <name> wrote:" possibly wrapped across lines ("...wrote:" alone).
    re.compile(r"^\s*On\b.*\bwrote:\s*$", re.IGNORECASE),
    re.compile(r"^\s*On\b.*", re.IGNORECASE),  # only if next lines are quoted (checked below)
    # A quoted header block: a line "From: ..." followed shortly by Sent/To/Subject.
    re.compile(r"^\s*From:\s.+", re.IGNORECASE),
]
_QUOTED_LINE = re.compile(r"^\s*>")
_HEADER_FOLLOWUP = re.compile(r"^\s*(Sent|To|Date|Subject|Cc):\s", re.IGNORECASE)

# --- Signature: standard "-- " delimiter or a sign-off line near the end.
_SIG_DELIM = re.compile(r"^--\s*$")
_SIGN_OFF = re.compile(
    r"^\s*(regards|best regards|kind regards|warm regards|best|thanks|"
    r"thank you|cheers|sincerely|respectfully|yours truly|talk soon)\s*[,!.]?\s*$",
    re.IGNORECASE,
)

# --- Disclaimer / confidentiality boilerplate: block starts at first match.
_DISCLAIMER = re.compile(
    r"^\s*(this (e-?mail|message|communication|transmission)\b"
    r"|the information (contained|in this)"
    r"|confidential(ity)?\b"
    r"|notice:|disclaimer:|privileged and confidential"
    r"|if you (are not|have received) )",
    re.IGNORECASE,
)


def normalize_body(body: str) -> str:
    """Light normalization: strip HTML if present, unify newlines, trim trailing ws.

    Only strips HTML when the payload actually looks like markup; plain-text
    bodies (the common Enron case) pass through untouched aside from newline
    normalization.
    """
    if not body:
        return ""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    if re.search(r"<\s*(html|body|div|p|br|table)\b", text, re.IGNORECASE):
        try:
            from bs4 import BeautifulSoup  # dep already installed (beautifulsoup4)

            soup = BeautifulSoup(text, "html.parser")
            for br in soup.find_all(["br"]):
                br.replace_with("\n")
            text = soup.get_text("\n")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        except Exception:  # ponytail: bs4 failure -> keep raw text, don't crash ingest
            pass
    # collapse 3+ blank lines to a single blank line; strip trailing spaces per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def _find_quote_start(lines: list[str]) -> int | None:
    """Return the line index where quoted history begins, or None."""
    for idx, line in enumerate(lines):
        if _QUOTED_LINE.match(line):
            return idx
        if _QUOTE_MARKERS[0].match(line) or _QUOTE_MARKERS[1].match(line):
            return idx
        if _QUOTE_MARKERS[2].match(line):  # "...wrote:" terminal
            return idx
        if _QUOTE_MARKERS[4].match(line):  # "From: ..." header block
            # Confirm it's a header block, not authored prose mentioning "From:".
            lookahead = "\n".join(lines[idx + 1 : idx + 5])
            if _HEADER_FOLLOWUP.search(lookahead) or _QUOTED_LINE.search(lookahead):
                return idx
        if _QUOTE_MARKERS[3].match(line):  # "On ..." possibly attribution
            lookahead = "\n".join(lines[idx : idx + 3])
            if re.search(r"\bwrote:", lookahead, re.IGNORECASE):
                return idx
    return None


def _find_signature_start(lines: list[str]) -> int | None:
    """Return the line index where the signature begins, or None. Conservative."""
    for idx, line in enumerate(lines):
        if _SIG_DELIM.match(line):
            return idx
    # Sign-off only counts when it sits in the tail of the message, so we do not
    # amputate a "Thanks," that is genuinely part of the authored body.
    tail_start = max(0, len(lines) - 6)
    for idx in range(tail_start, len(lines)):
        if _SIGN_OFF.match(lines[idx]):
            return idx
    return None


def _find_disclaimer_start(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if _DISCLAIMER.match(line):
            return idx
    return None


def segment_email_body(body: str) -> EmailSegments:
    """Split an email body into authored / quoted / signature / disclaimer.

    The authored portion is what the normal retrieval path uses. Segmentation is
    deterministic and conservative: when in doubt, content stays authored.
    """
    normalized = normalize_body(body)
    lines = normalized.split("\n")

    # 1. Peel off quoted reply history (everything after the first quote marker).
    quote_idx = _find_quote_start(lines)
    if quote_idx is not None:
        head_lines = lines[:quote_idx]
        quoted_text = "\n".join(lines[quote_idx:]).strip()
    else:
        head_lines = lines
        quoted_text = ""

    # 2. Peel off a trailing disclaimer block from the head.
    disc_idx = _find_disclaimer_start(head_lines)
    if disc_idx is not None:
        disclaimer_text = "\n".join(head_lines[disc_idx:]).strip()
        head_lines = head_lines[:disc_idx]
    else:
        disclaimer_text = ""

    # 3. Peel off the signature from what remains.
    sig_idx = _find_signature_start(head_lines)
    if sig_idx is not None:
        signature_text = "\n".join(head_lines[sig_idx:]).strip()
        authored_lines = head_lines[:sig_idx]
    else:
        signature_text = ""
        authored_lines = head_lines

    authored_text = "\n".join(authored_lines).strip()

    # Conservative fallback: only restore content when signature/disclaimer
    # stripping over-ate a body that had no quote block (nothing genuinely
    # quoted to protect against). A quote-only email (authored_lines empty
    # because everything after line 0 was quoted history) must legitimately
    # produce empty authored_text -- resurrecting the quote here would put
    # reply history straight into the retrieval path, which is exactly what
    # segmentation exists to prevent.
    if not authored_text and normalized and quote_idx is None:
        authored_text = normalized
        signature_text = disclaimer_text = ""

    # Offsets of authored_text within normalized (for citation provenance).
    start = normalized.find(authored_text) if authored_text else 0
    if start < 0:
        start = 0
    end = start + len(authored_text)

    return EmailSegments(
        authored_text=authored_text,
        quoted_text=quoted_text,
        signature_text=signature_text,
        disclaimer_text=disclaimer_text,
        normalized=normalized,
        authored_start=start,
        authored_end=end,
    )


def _format_date(value) -> str:
    """Deterministic YYYY-MM-DD from a datetime/date, empty string if absent."""
    if value is None:
        return ""
    try:
        return value.date().isoformat()  # datetime
    except AttributeError:
        try:
            return value.isoformat()  # date
        except AttributeError:
            return str(value)


def build_embed_text(
    text: str,
    *,
    sender: str | None = None,
    to: list[str] | None = None,
    cc: list[str] | None = None,
    date=None,
    subject: str | None = None,
    thread_id: str | None = None,
    in_reply_to: str | None = None,
    context_prefix: str | None = None,
) -> str:
    """Build compact retrieval text: header block + optional context + exact ``text``.

    This is the ONLY place headers/context are assembled into retrieval text.
    Stage 4 re-runs it with a ``context_prefix``; every other caller passes none
    and gets byte-identical Stage-1 output, which is what lets a chunk be
    re-contextualized without its deterministic form ever drifting.

    Only ``embed_text`` carries headers and context; the citable ``text`` stays
    pure authored evidence. ``Cc`` is populated strictly from ``cc`` (never
    ``to``). Absent fields are omitted rather than rendered blank.
    """
    header_lines: list[str] = []
    if sender:
        header_lines.append(f"From: {sender.strip()}")
    if to:
        header_lines.append(f"To: {', '.join(a for a in to if a)}")
    if cc:
        header_lines.append(f"Cc: {', '.join(a for a in cc if a)}")
    date_str = _format_date(date)
    if date_str:
        header_lines.append(f"Date: {date_str}")
    if subject:
        header_lines.append(f"Subject: {subject.strip()}")
    if thread_id:
        header_lines.append(f"Thread-ID: {thread_id}")
    # Deterministic thread context only: parent id, never a fabricated summary.
    if in_reply_to:
        header_lines.append(f"In-Reply-To: {in_reply_to}")

    body = text
    if context_prefix and context_prefix.strip():
        # Between headers and evidence, never inside `text`: the model's words
        # are retrieval scaffolding and must never become citable evidence.
        body = f"{context_prefix.strip()}\n\n{text}"

    if not header_lines:
        return body
    return "\n".join(header_lines) + "\n\n" + body


# Known limitations of deterministic segmentation (documented, not hidden):
#   * Bottom-posted quoting without ">" or a recognized marker is not detected.
#   * Signatures without a "-- " delimiter or a recognized sign-off line survive
#     into the authored body (conservative bias: keep content).
#   * A disclaimer that is not on its own line, or in a language/wording outside
#     the marker set, is not stripped.
#   * HTML normalization is best-effort; complex nested markup may leave noise.


def _demo() -> None:
    raw = (
        "Hi Sarah,\n\n"
        "The approved budget is now $120,000. Please proceed.\n\n"
        "Thanks,\n"
        "Bob\n"
        "--\n"
        "Bob Smith | Finance | bob@acme.com\n\n"
        "This e-mail is confidential and intended only for the addressee.\n\n"
        "On Tue, May 2, Sarah wrote:\n"
        "> What is the final number?\n"
    )
    seg = segment_email_body(raw)
    assert "approved budget is now $120,000" in seg.authored_text, seg.authored_text
    assert "Bob Smith" not in seg.authored_text, "signature leaked into authored"
    assert "confidential" not in seg.authored_text, "disclaimer leaked into authored"
    assert "What is the final number" not in seg.authored_text, "quote leaked"
    assert "What is the final number" in seg.quoted_text
    assert seg.normalized[seg.authored_start : seg.authored_end] == seg.authored_text

    embed = build_embed_text(
        seg.authored_text,
        sender="Bob <bob@acme.com>",
        to=["sarah@acme.com"],
        cc=["finance@acme.com"],
        subject="Q3 Budget",
        thread_id="thread_123",
    )
    assert "Cc: finance@acme.com" in embed
    assert "To: sarah@acme.com" in embed
    assert embed.endswith(seg.authored_text)

    # Quote-only email (pure forward, no new words) must yield empty authored
    # text, not the resurrected quoted body.
    quote_only = "On Tue, May 2, Sarah wrote:\n> What is the final approved number?\n> Thanks\n"
    quote_only_seg = segment_email_body(quote_only)
    assert quote_only_seg.authored_text == "", quote_only_seg.authored_text
    assert "final approved number" in quote_only_seg.quoted_text

    print("email_segmentation self-check OK")


if __name__ == "__main__":
    _demo()
