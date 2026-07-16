from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from email_thread_rag.config import get_settings
from email_thread_rag.rag.mailbox_slice import load_mailbox_messages, select_mailbox_messages, write_selected_message
from email_thread_rag.rag.utils import read_json, write_json


MAILBOX_JSON_RE = re.compile(r"^(index|mailbox(?:_part_(\d+))?)\.json$")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlsplit(url)
    encoded_url = urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            urllib.parse.quote(parsed.path, safe="/%"),
            parsed.query,
            parsed.fragment,
        )
    )
    with urllib.request.urlopen(encoded_url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _copy_local(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _download_hf(repo_id: str, repo_type: str, revision: str, repo_path: str, destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=repo_path,
    )
    _copy_local(Path(cached_path), destination)


def _github_api_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "email-thread-rag"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _download_github(repo: str, revision: str, repo_path: str, destination: Path) -> None:
    metadata = _github_api_json(
        f"https://api.github.com/repos/{repo}/contents/{repo_path}?ref={revision}"
    )
    download_url = metadata.get("download_url")
    if not download_url:
        raise ValueError(f"Could not resolve a download URL for {repo_path} in {repo}@{revision}")
    _download(download_url, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_file(base_dir: Path, file_meta: dict, force: bool) -> dict:
    relative_path = file_meta["relative_path"]
    destination = base_dir / relative_path
    if force and destination.exists():
        destination.unlink()

    if not destination.exists():
        if "local_source" in file_meta:
            _copy_local(Path(file_meta["local_source"]), destination)
        elif "gh_repo" in file_meta and "repo_path" in file_meta:
            _download_github(
                repo=file_meta["gh_repo"],
                revision=file_meta.get("gh_revision", "main"),
                repo_path=file_meta["repo_path"],
                destination=destination,
            )
        elif "hf_repo_id" in file_meta and "repo_path" in file_meta:
            _download_hf(
                repo_id=file_meta["hf_repo_id"],
                repo_type=file_meta.get("hf_repo_type", "dataset"),
                revision=file_meta.get("hf_revision", "main"),
                repo_path=file_meta["repo_path"],
                destination=destination,
            )
        else:
            _download(file_meta["url"], destination)

    actual_sha = _sha256(destination)
    expected_sha = file_meta.get("sha256")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"Checksum mismatch for {destination}: expected {expected_sha}, got {actual_sha}")

    return {
        **file_meta,
        "local_path": relative_path,
        "actual_sha256": actual_sha,
    }


def _mailbox_sort_key(repo_path: str) -> tuple[int, int]:
    name = Path(repo_path).name
    match = MAILBOX_JSON_RE.match(name)
    if not match:
        return (2, 0)
    if name == "index.json":
        return (0, 0)
    if name == "mailbox.json":
        return (1, 0)
    return (2, int(match.group(2) or 0))


def _discover_hf_mailbox_repo_paths(source_meta: dict[str, Any], mailbox: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    entries = api.list_repo_tree(
        repo_id=source_meta.get("repo_id"),
        repo_type=source_meta.get("repo_type", "dataset"),
        revision=source_meta.get("revision", "main"),
        path_in_repo=f"mail/{mailbox}",
        recursive=False,
    )
    repo_paths: list[str] = []
    for entry in entries:
        repo_path = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
        if not repo_path:
            continue
        if MAILBOX_JSON_RE.match(Path(repo_path).name):
            repo_paths.append(repo_path)
    repo_paths.sort(key=_mailbox_sort_key)
    if not repo_paths:
        raise ValueError(f"No mailbox JSON files found in enronarchive/mail for mailbox '{mailbox}'.")
    return repo_paths


def _discover_github_mailbox_repo_paths(source_meta: dict[str, Any], mailbox: str) -> list[str]:
    repo = source_meta.get("repo")
    revision = source_meta.get("revision", "main")
    payload = _github_api_json(
        f"https://api.github.com/repos/{repo}/contents/mail/{mailbox}?ref={revision}"
    )
    repo_paths: list[str] = []
    if isinstance(payload, list):
        for entry in payload:
            repo_path = entry.get("path")
            if not repo_path:
                continue
            if MAILBOX_JSON_RE.match(Path(repo_path).name):
                repo_paths.append(repo_path)
    repo_paths.sort(key=_mailbox_sort_key)
    if not repo_paths:
        raise ValueError(f"No mailbox JSON files found in {repo} for mailbox '{mailbox}'.")
    return repo_paths


def _discover_mailbox_repo_paths(source_meta: dict[str, Any], mailbox: str) -> list[str]:
    if source_meta.get("provider") == "github_repo":
        return _discover_github_mailbox_repo_paths(source_meta, mailbox)
    return _discover_hf_mailbox_repo_paths(source_meta, mailbox)


def _augment_source_file_meta(source_meta: dict[str, Any], file_meta: dict[str, Any]) -> dict[str, Any]:
    if source_meta.get("provider") == "github_repo":
        return {
            **file_meta,
            "gh_repo": source_meta.get("repo"),
            "gh_revision": source_meta.get("revision", "main"),
        }
    return {
        **file_meta,
        "hf_repo_id": source_meta.get("repo_id"),
        "hf_repo_type": source_meta.get("repo_type", "dataset"),
        "hf_revision": source_meta.get("revision", "main"),
    }


def _download_discovered_mailbox_files(settings, source_meta: dict[str, Any], mailbox: str, force: bool) -> list[dict]:
    resolved: list[dict] = []
    for repo_path in _discover_mailbox_repo_paths(source_meta, mailbox):
        resolved.append(
            _materialize_file(
                settings.raw_data_dir,
                _augment_source_file_meta(
                    source_meta,
                    {
                    "relative_path": str(Path("_mailboxes") / mailbox / Path(repo_path).name),
                    "repo_path": repo_path,
                    },
                ),
                force,
            )
        )
    return resolved


def _is_remote_entry_not_found(exc: Exception) -> bool:
    if exc.__class__.__name__ in {"RemoteEntryNotFoundError", "EntryNotFoundError"}:
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
        return True
    return False


def _resolve_mailbox_source_files(settings, source_meta: dict[str, Any], mailbox_meta: dict[str, Any], force: bool) -> list[dict]:
    mailbox = mailbox_meta["mailbox"]
    repo_patterns = mailbox_meta.get("repo_patterns")
    if repo_patterns:
        from huggingface_hub import snapshot_download

        snapshot_root = Path(
            snapshot_download(
                repo_id=source_meta.get("repo_id"),
                repo_type=source_meta.get("repo_type", "dataset"),
                revision=source_meta.get("revision", "main"),
                allow_patterns=repo_patterns,
            )
        )
        resolved = []
        for matched_path in sorted({path for pattern in repo_patterns for path in snapshot_root.glob(pattern)}):
            relative_path = matched_path.relative_to(snapshot_root)
            destination_relative_path = Path("_mailboxes") / mailbox / relative_path.name
            resolved.append(
                _materialize_file(
                    settings.raw_data_dir,
                    {
                        "relative_path": str(destination_relative_path),
                        "local_source": str(matched_path),
                    },
                    force,
                )
            )
        if resolved:
            return resolved

    source_files = mailbox_meta.get("files")
    if source_files:
        resolved = []
        try:
            for index, file_meta in enumerate(source_files, start=1):
                relative_path = file_meta.get("relative_path") or f"_mailboxes/{mailbox}/mailbox_part_{index}.json"
                resolved.append(
                    _materialize_file(
                        settings.raw_data_dir,
                        _augment_source_file_meta(
                            source_meta,
                            {
                            **file_meta,
                            "relative_path": relative_path,
                            },
                        ),
                        force,
                    )
                )
        except Exception as exc:
            if not _is_remote_entry_not_found(exc):
                raise
            repo_path = source_files[0].get("repo_path", "") if len(source_files) == 1 else ""
            if len(source_files) == 1 and (repo_path.endswith("/mailbox.json") or repo_path.endswith("/index.json")):
                return _download_discovered_mailbox_files(settings, source_meta, mailbox, force)
            raise exc
        return resolved

    # Default mailbox JSON fallback for Enron Archive mailbox slices.
    return _download_discovered_mailbox_files(settings, source_meta, mailbox, force)


def _resolve_attachment_file(
    settings,
    source_meta: dict[str, Any],
    mailbox: str,
    message: dict[str, Any],
    attachment: dict[str, Any],
    index: int,
    force: bool,
) -> dict:
    filename = attachment["filename"]
    attachment_relative_path = (
        f"_attachments/{mailbox}/{message['doc_id']}/{index:02d}_{filename}"
    )
    attachment_meta = {
        "filename": filename,
        "relative_path": attachment_relative_path,
        "sha256": attachment.get("sha256"),
    }
    if attachment.get("local_source"):
        attachment_meta["local_source"] = attachment["local_source"]
    elif attachment.get("repo_path"):
        repo_path = str(attachment["repo_path"])
        if not repo_path.startswith("mail/"):
            if repo_path.startswith("attachments/"):
                repo_path = f"mail/{mailbox}/{repo_path}"
            elif repo_path.startswith(f"{mailbox}/"):
                repo_path = f"mail/{repo_path}"
        attachment_base_url = source_meta.get("attachment_base_url")
        if attachment_base_url and "/attachments/" in repo_path:
            attachment_meta["url"] = f"{attachment_base_url.rstrip('/')}/{repo_path}"
        else:
            attachment_meta["repo_path"] = repo_path
            attachment_meta = _augment_source_file_meta(source_meta, attachment_meta)
    elif attachment.get("url"):
        attachment_meta["url"] = attachment["url"]
    else:
        raise ValueError(
            f"Attachment source missing for {message['message_id']} attachment {filename}. "
            "Expected local_source, repo_path, or url."
        )
    resolved_attachment = _materialize_file(settings.raw_data_dir, attachment_meta, force)
    return {
        "attachment_id": attachment.get("attachment_id"),
        "filename": filename,
        "relative_path": resolved_attachment["local_path"],
    }


def _build_mailbox_slice(settings, manifest: dict[str, Any], force: bool) -> dict[str, Any]:
    source_meta = manifest.get("source", {})
    resolved_threads: list[dict[str, Any]] = []
    message_count = 0
    attachment_count = 0

    for mailbox_meta in manifest.get("mailboxes", []):
        mailbox = mailbox_meta["mailbox"]
        resolved_mailbox_files = _resolve_mailbox_source_files(settings, source_meta, mailbox_meta, force)
        mailbox_paths = [settings.raw_data_dir / file_meta["local_path"] for file_meta in resolved_mailbox_files]
        mailbox_messages = load_mailbox_messages(mailbox_paths)
        selected_messages = select_mailbox_messages(mailbox, mailbox_messages, mailbox_meta.get("selection", {}))

        thread_groups: dict[str, list[dict[str, Any]]] = {}
        for message in selected_messages:
            thread_groups.setdefault(message["thread_id"], []).append(message)

        for thread_id, messages in sorted(thread_groups.items()):
            resolved_messages = []
            for message in sorted(messages, key=lambda item: (item["date"], item["message_id"])):
                resolved_attachments = []
                for index, attachment in enumerate(message.get("attachments", []), start=1):
                    resolved_attachments.append(
                        _resolve_attachment_file(settings, source_meta, mailbox, message, attachment, index, force)
                    )
                    attachment_count += 1

                message_payload = {**message, "attachments": resolved_attachments}
                relative_path = f"_selected/{mailbox}/{message['doc_id']}.json"
                write_selected_message(message_payload, settings.raw_data_dir / relative_path)
                resolved_messages.append(
                    {
                        "message_id": message["message_id"],
                        "relative_path": relative_path,
                        "local_path": relative_path,
                        "actual_sha256": _sha256(settings.raw_data_dir / relative_path),
                        "attachments": [
                            {
                                "filename": attachment["filename"],
                                "relative_path": attachment["relative_path"],
                                "local_path": attachment["relative_path"],
                                "actual_sha256": _sha256(settings.raw_data_dir / attachment["relative_path"]),
                            }
                            for attachment in resolved_attachments
                        ],
                    }
                )
                message_count += 1

            resolved_threads.append(
                {
                    "thread_key": thread_id,
                    "mailbox": mailbox,
                    "messages": resolved_messages,
                }
            )

    return {
        **manifest,
        "threads": resolved_threads,
        "resolved_counts": {
            "thread_count": len(resolved_threads),
            "message_count": message_count,
            "attachment_count": attachment_count,
        },
    }


def build_dataset_slice(force: bool = False) -> dict:
    settings = get_settings()
    manifest = read_json(settings.dataset_manifest_path)
    if manifest.get("mailboxes"):
        resolved_manifest = _build_mailbox_slice(settings, manifest, force)
        write_json(settings.resolved_manifest_path, resolved_manifest)
        return resolved_manifest

    resolved_threads = []
    message_count = 0
    attachment_count = 0

    for thread in manifest.get("threads", []):
        resolved_messages = []
        for message in thread.get("messages", []):
            resolved_message = _materialize_file(settings.raw_data_dir, message, force)
            resolved_attachments = []
            for attachment in message.get("attachments", []):
                resolved_attachment = _materialize_file(settings.raw_data_dir, attachment, force)
                resolved_attachments.append(resolved_attachment)
                attachment_count += 1
            resolved_message["attachments"] = resolved_attachments
            resolved_messages.append(resolved_message)
            message_count += 1
        resolved_threads.append({**thread, "messages": resolved_messages})

    resolved_manifest = {
        **manifest,
        "threads": resolved_threads,
        "resolved_counts": {
            "thread_count": len(resolved_threads),
            "message_count": message_count,
            "attachment_count": attachment_count,
        },
    }
    write_json(settings.resolved_manifest_path, resolved_manifest)
    return resolved_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pinned email dataset slice.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if present locally.")
    args = parser.parse_args()
    resolved = build_dataset_slice(force=args.force)
    print(json.dumps(resolved.get("resolved_counts", {}), indent=2))


if __name__ == "__main__":
    main()
