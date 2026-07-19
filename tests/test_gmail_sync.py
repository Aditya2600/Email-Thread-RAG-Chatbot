"""Worker + cursor state machine, driven entirely by FakeGmailClient.

No network, no credentials, no Postgres: InMemorySyncStore + InMemoryChunkSink
stand in for the durable pair, which tests/integration/test_gmail_paradedb.py
exercises for real.
"""

from __future__ import annotations

import pytest

from email_thread_rag.gmail.fakes import FakeGmailClient, build_gmail_message
from email_thread_rag.gmail.sink import InMemoryChunkSink
from email_thread_rag.gmail.store import InMemorySyncStore
from email_thread_rag.gmail.worker import SyncWorker
from gmail_store_contract import connect_mailbox, notify

ADDRESS = "user@example.com"


def message(gmail_id: str, *, history_id: int = 200, body: str = "The approved amount is $1200.", thread="t-1"):
    return build_gmail_message(
        gmail_id=gmail_id,
        thread_id=thread,
        history_id=history_id,
        sender="alice@corp.com",
        to="bob@corp.com",
        subject="Budget Review",
        body=body,
    )


def history_page(*, history_id, added=(), deleted=(), next_page_token=None):
    records = []
    for record_id, gmail_id in added:
        records.append({"id": str(record_id), "messagesAdded": [{"message": {"id": gmail_id}}]})
    for record_id, gmail_id in deleted:
        records.append({"id": str(record_id), "messagesDeleted": [{"message": {"id": gmail_id}}]})
    page = {"historyId": str(history_id), "history": records}
    if next_page_token:
        page["nextPageToken"] = next_page_token
    return page


@pytest.fixture
def store():
    return InMemorySyncStore()


@pytest.fixture
def sink():
    return InMemoryChunkSink()


def build_worker(store, sink, client):
    return SyncWorker(store, client_factory=lambda _m: client, sink_factory=lambda _m: sink)


# --- Delta sync -----------------------------------------------------------
def test_worker_starts_history_from_the_committed_cursor(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    notify(store, ADDRESS, 300, "pubsub-1")

    client = FakeGmailClient(email_address=ADDRESS, history_pages=[history_page(history_id=300)])
    build_worker(store, sink, client).run_once()

    # Not the notification's historyId, not the watch's: the committed cursor.
    assert client.history_calls() == [250]


def test_successful_sync_ingests_canonical_chunks_and_advances_the_cursor(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    notify(store, ADDRESS, 200, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[history_page(history_id=200, added=[(150, "m-1")])],
    )
    outcome = build_worker(store, sink, client).run_once()

    assert outcome.added == 1
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 200

    chunks = sink.all_chunks()
    assert chunks and all(chunk.message_id == "m-1" for chunk in chunks)
    # Canonical Stage-1 output: exact authored text as evidence, headers only in
    # the retrieval-side embed_text.
    assert "The approved amount is $1200." in chunks[0].text
    assert "Subject: Budget Review" not in chunks[0].text
    assert "Budget Review" in chunks[0].embed_text
    assert chunks[0].source_type == "gmail"
    assert chunks[0].thread_id == "t-1"


def test_history_pages_are_deduplicated(store, sink):
    connect_mailbox(store, address=ADDRESS, history_id=100)
    notify(store, ADDRESS, 400, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[
            history_page(history_id=300, added=[(150, "m-1")], next_page_token="page-2"),
            history_page(history_id=400, added=[(160, "m-1")]),
        ],
    )
    outcome = build_worker(store, sink, client).run_once()

    # The same message on two pages is fetched and persisted once.
    assert outcome.added == 1
    assert [c for c in client.calls if c[0] == "get_message"] == [("get_message", "m-1")]
    assert outcome.committed_history_id == 400


def test_deleted_message_is_removed_from_the_index(store, sink):
    connect_mailbox(store, address=ADDRESS, history_id=100)
    notify(store, ADDRESS, 200, "pubsub-1")
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[history_page(history_id=200, added=[(150, "m-1")])],
    )
    worker = build_worker(store, sink, client)
    worker.run_once()
    assert sink.chunks_by_message["m-1"]

    client.history_pages = [history_page(history_id=300, deleted=[(250, "m-1")])]
    notify(store, ADDRESS, 300, "pubsub-2")
    worker.run_once()

    assert "m-1" not in sink.chunks_by_message
    assert sink.all_chunks() == []


def test_added_then_deleted_in_one_window_leaves_nothing_indexed(store, sink):
    connect_mailbox(store, address=ADDRESS, history_id=100)
    notify(store, ADDRESS, 300, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[history_page(history_id=300, added=[(150, "m-1")], deleted=[(250, "m-1")])],
    )
    build_worker(store, sink, client).run_once()

    assert sink.all_chunks() == []


def test_message_vanishing_before_fetch_is_not_an_error(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    notify(store, ADDRESS, 200, "pubsub-1")

    # messages.get 404s -> FakeGmailClient returns None, same as the real one.
    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={},
        history_pages=[history_page(history_id=200, added=[(150, "m-gone")])],
    )
    outcome = build_worker(store, sink, client).run_once()

    assert outcome.added == 0
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 200


# --- Failure handling -----------------------------------------------------
def test_failed_processing_does_not_advance_the_cursor(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    job = notify(store, ADDRESS, 400, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[history_page(history_id=400, added=[(300, "m-1")])],
    )
    client.fail_get_message_ids = {"m-1"}

    assert build_worker(store, sink, client).run_once() is None
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 250
    assert store.get_job(job.id).status == "pending"  # retryable
    assert sink.all_chunks() == []


def test_retry_after_a_failure_succeeds_from_the_same_cursor(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    notify(store, ADDRESS, 400, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[history_page(history_id=400, added=[(300, "m-1")])],
    )
    client.fail_get_message_ids = {"m-1"}
    worker = build_worker(store, sink, client)
    worker.run_once()

    client.fail_get_message_ids = set()
    outcome = worker.run_once()

    assert outcome.added == 1
    assert client.history_calls() == [250, 250]  # the retry re-read the same window
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 400


def test_sink_failure_leaves_the_cursor_intact(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    notify(store, ADDRESS, 400, "pubsub-1")

    class BrokenSink(InMemoryChunkSink):
        def persist(self, email):
            raise RuntimeError("paradedb is down")

    client = FakeGmailClient(
        email_address=ADDRESS,
        messages={"m-1": message("m-1")},
        history_pages=[history_page(history_id=400, added=[(300, "m-1")])],
    )
    SyncWorker(store, client_factory=lambda _m: client, sink_factory=lambda _m: BrokenSink()).run_once()

    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 250
    assert store.get_mailbox_by_db_id(mailbox.id).status == "error"


# --- History expiry / full sync ------------------------------------------
def test_history_404_triggers_a_full_sync_without_advancing_the_old_cursor(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    notify(store, ADDRESS, 900, "pubsub-1")

    client = FakeGmailClient(
        email_address=ADDRESS,
        profile_history_id=800,
        messages={"m-1": message("m-1"), "m-2": message("m-2", body="Second mail.")},
    )
    client.history_expired = True

    outcome = build_worker(store, sink, client).run_once()

    assert outcome.full_sync is True
    assert outcome.added == 2
    assert sorted(sink.chunks_by_message) == ["m-1", "m-2"]
    # The new cursor is the checkpoint taken before the scan -- never the stale
    # one, and never a value the replay did not confirm.
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 800


def test_full_sync_takes_its_checkpoint_before_scanning(store, sink):
    """The checkpoint must precede messages.list, or mail arriving during the
    scan falls into a gap that the replay would never cover."""
    connect_mailbox(store, address=ADDRESS, history_id=100)
    notify(store, ADDRESS, 900, "pubsub-1")

    client = FakeGmailClient(email_address=ADDRESS, profile_history_id=800, messages={"m-1": message("m-1")})
    client.history_expired = True
    build_worker(store, sink, client).run_once()

    call_names = [name for name, _ in client.calls]
    assert call_names.index("get_profile") < call_names.index("list_messages")


def test_full_sync_replays_history_since_the_checkpoint(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    notify(store, ADDRESS, 900, "pubsub-1")

    class ExpireOnceClient(FakeGmailClient):
        """404s the stale cursor, then serves the replay from the checkpoint --
        the sequence a real mailbox with an aged-out cursor produces.

        m-late models mail that lands *during* the scan: messages.list never
        returns it, so only the post-checkpoint replay can find it.
        """

        def list_history(self, *, start_history_id, history_types=None):
            self.calls.append(("list_history", start_history_id))
            if start_history_id == 250:
                from email_thread_rag.gmail.client import GmailHistoryExpired

                raise GmailHistoryExpired("too old")
            yield history_page(history_id=850, added=[(820, "m-late")])

        def list_messages(self, *, query=None):
            self.calls.append(("list_messages", query))
            yield {"messages": [{"id": "m-1"}]}

    client = ExpireOnceClient(
        email_address=ADDRESS,
        profile_history_id=800,
        messages={"m-1": message("m-1"), "m-late": message("m-late", body="Arrived during the scan.")},
    )

    outcome = build_worker(store, sink, client).run_once()

    assert client.history_calls() == [250, 800]
    assert "m-late" in sink.chunks_by_message
    # Cursor is the replay's end, not the checkpoint.
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 850
    assert outcome.full_sync is True


def test_full_sync_is_idempotent(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)

    client = FakeGmailClient(email_address=ADDRESS, profile_history_id=800, messages={"m-1": message("m-1")})
    client.history_expired = True
    worker = build_worker(store, sink, client)

    notify(store, ADDRESS, 900, "pubsub-1")
    worker.run_once()
    first = {mid: len(chunks) for mid, chunks in sink.chunks_by_message.items()}

    # Force a second full sync (as a crashed-and-retried run would do).
    store.set_mailbox_status(mailbox.id, "needs_full_sync")
    notify(store, ADDRESS, 950, "pubsub-2")
    worker.run_once()

    assert {mid: len(chunks) for mid, chunks in sink.chunks_by_message.items()} == first


def test_history_expiry_marks_the_mailbox_before_rebuilding(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS, history_id=100)
    store.commit_history_cursor(mailbox.id, 250)
    job = notify(store, ADDRESS, 900, "pubsub-1")

    class FailAfterMarkClient(FakeGmailClient):
        def get_profile(self):
            # Crash right after the 404 mark, before any rebuild work.
            raise RuntimeError("crash during full sync")

    client = FailAfterMarkClient(email_address=ADDRESS)
    client.history_expired = True
    build_worker(store, sink, client).run_once()

    # The marks survive the crash, so the retry is a full sync from a cursor
    # that never moved.
    assert store.get_job(job.id).needs_full_sync is True
    assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 250


def test_mailbox_with_no_cursor_does_a_full_sync(store, sink):
    record = store.upsert_mailbox(
        tenant_id="acme",
        mailbox_id="inbox",
        email_address=ADDRESS,
        refresh_token_ciphertext=b"x",
        token_key_id="k",
    )
    notify(store, ADDRESS, 500, "pubsub-1")

    client = FakeGmailClient(email_address=ADDRESS, profile_history_id=500, messages={"m-1": message("m-1")})
    outcome = build_worker(store, sink, client).run_once()

    assert outcome.full_sync is True
    assert store.get_mailbox_by_db_id(record.id).last_committed_history_id == 500


# --- Worker plumbing ------------------------------------------------------
def test_worker_returns_none_on_an_empty_queue(store, sink):
    assert build_worker(store, sink, FakeGmailClient()).run_once() is None


def test_job_for_a_disconnected_mailbox_is_dropped(store, sink):
    mailbox = connect_mailbox(store, address=ADDRESS)
    notify(store, ADDRESS, 200, "pubsub-1")
    store.disconnect_mailbox(mailbox.id)

    client = FakeGmailClient(email_address=ADDRESS)
    assert build_worker(store, sink, client).run_once() is None
    assert client.calls == []  # no Gmail call for a disconnected mailbox


def test_tenant_isolation_across_two_mailboxes(store):
    acme = connect_mailbox(store, tenant="acme", mailbox="inbox", address="a@example.com", history_id=100)
    globex = connect_mailbox(store, tenant="globex", mailbox="inbox", address="b@example.com", history_id=100)

    sinks = {acme.id: InMemoryChunkSink(), globex.id: InMemoryChunkSink()}
    clients = {
        acme.id: FakeGmailClient(
            email_address="a@example.com",
            messages={"m-acme": message("m-acme", body="Acme only.")},
            history_pages=[history_page(history_id=200, added=[(150, "m-acme")])],
        ),
        globex.id: FakeGmailClient(
            email_address="b@example.com",
            messages={"m-globex": message("m-globex", body="Globex only.")},
            history_pages=[history_page(history_id=300, added=[(250, "m-globex")])],
        ),
    }
    worker = SyncWorker(
        store,
        client_factory=lambda m: clients[m.id],
        sink_factory=lambda m: sinks[m.id],
    )

    notify(store, "a@example.com", 200, "pubsub-acme")
    notify(store, "b@example.com", 300, "pubsub-globex")
    worker.drain()

    assert list(sinks[acme.id].chunks_by_message) == ["m-acme"]
    assert list(sinks[globex.id].chunks_by_message) == ["m-globex"]
    assert store.get_mailbox_by_db_id(acme.id).last_committed_history_id == 200
    assert store.get_mailbox_by_db_id(globex.id).last_committed_history_id == 300
