from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from email_thread_rag.rag.parse_attachments import parse_doc_attachment, parse_rtf_attachment, parse_xls_attachment


def test_parse_rtf_attachment(tmp_path):
    path = tmp_path / "note.rtf"
    path.write_text(r"{\rtf1\ansi This is an RTF amount \$3210 approval.}", encoding="utf-8")
    record = parse_rtf_attachment(
        path,
        attachment_id="rtf-att-1",
        message_id="<rtf@example.com>",
        thread_id="thread-rtf",
    )
    assert "3210" in record.pages[0].text


def test_parse_doc_attachment_uses_antiword(monkeypatch, tmp_path):
    path = tmp_path / "memo.doc"
    path.write_bytes(b"fake-doc")

    monkeypatch.setattr("email_thread_rag.rag.parse_attachments.shutil.which", lambda name: "/usr/bin/antiword" if name == "antiword" else None)
    monkeypatch.setattr(
        "email_thread_rag.rag.parse_attachments.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Approved amount $5555", returncode=0),
    )

    record = parse_doc_attachment(
        path,
        attachment_id="doc-att-1",
        message_id="<doc@example.com>",
        thread_id="thread-doc",
    )
    assert "$5555" in record.pages[0].text


def test_parse_xls_attachment_reads_sheet_values(monkeypatch, tmp_path):
    path = tmp_path / "sheet.xls"
    path.write_bytes(b"fake-xls")

    class FakeSheet:
        name = "Budget"
        nrows = 2
        ncols = 2

        def cell_value(self, row_idx, col_idx):
            values = {
                (0, 0): "Vendor",
                (0, 1): "Amount",
                (1, 0): "Acme",
                (1, 1): "1500",
            }
            return values[(row_idx, col_idx)]

    class FakeWorkbook:
        def sheets(self):
            return [FakeSheet()]

        def release_resources(self):
            return None

    monkeypatch.setattr("email_thread_rag.rag.parse_attachments.xlrd.open_workbook", lambda *args, **kwargs: FakeWorkbook())

    record = parse_xls_attachment(
        path,
        attachment_id="xls-att-1",
        message_id="<xls@example.com>",
        thread_id="thread-xls",
    )
    assert "Acme" in record.pages[0].text
    assert "1500" in record.pages[0].text
