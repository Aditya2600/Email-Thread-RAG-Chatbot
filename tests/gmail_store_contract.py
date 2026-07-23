"""Contract every SyncStore must satisfy.

Subclassed twice: by ``tests/test_gmail_store.py`` against
``InMemorySyncStore`` (no DB, runs in the default suite) and by
``tests/integration/test_gmail_paradedb.py`` against ``PostgresSyncStore``.
The point is that the in-memory store used by the fast tests is not allowed to
quietly disagree with the real one.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from email_thread_rag.gmail.models import GmailNotification, OAuthState
from email_thread_rag.gmail.store import utcnow

TOPIC = "projects/demo/topics/gmail-sync"


def connect_mailbox(store, *, tenant="acme", mailbox="inbox", address="user@example.com", history_id=100):
    record = store.upsert_mailbox(
        tenant_id=tenant,
        mailbox_id=mailbox,
        email_address=address,
        refresh_token_ciphertext=b"ciphertext-not-a-token",
        token_key_id="test-key",
    )
    return store.activate_watch(
        record.id, history_id=history_id, topic=TOPIC, expiration=utcnow() + timedelta(days=7)
    )


def notify(store, address, history_id, message_id):
    return store.record_notification(
        GmailNotification(email_address=address, history_id=history_id, pubsub_message_id=message_id)
    )


class SyncStoreContract:
    """Subclass and provide a ``store`` fixture."""

    # --- OAuth state ------------------------------------------------------
    def test_oauth_state_is_single_use(self, store):
        state = OAuthState(
            state="state-single",
            tenant_id="acme",
            mailbox_id="inbox",
            code_verifier="verifier",
            redirect_uri="https://app.example.com/oauth/callback",
            expires_at=utcnow() + timedelta(minutes=10),
        )
        store.create_oauth_state(state)

        first = store.consume_oauth_state("state-single")
        assert first is not None and first.code_verifier == "verifier"
        # A replayed callback must not be able to redeem the same state again.
        assert store.consume_oauth_state("state-single") is None

    def test_oauth_state_expires(self, store):
        store.create_oauth_state(
            OAuthState(
                state="state-expired",
                tenant_id="acme",
                mailbox_id="inbox",
                code_verifier="verifier",
                redirect_uri="https://app.example.com/oauth/callback",
                expires_at=utcnow() - timedelta(seconds=1),
            )
        )
        assert store.consume_oauth_state("state-expired") is None

    def test_unknown_oauth_state_is_rejected(self, store):
        assert store.consume_oauth_state("never-issued") is None

    # --- Watch / mailbox --------------------------------------------------
    def test_activate_watch_persists_cursor_and_marks_active(self, store):
        mailbox = connect_mailbox(store, history_id=4242)
        assert mailbox.status == "active"
        assert mailbox.last_committed_history_id == 4242
        assert mailbox.watch_expiration is not None

        reloaded = store.get_mailbox("acme", "inbox")
        assert reloaded.last_committed_history_id == 4242

    def test_watch_renewal_does_not_rewind_committed_cursor(self, store):
        mailbox = connect_mailbox(store, history_id=100)
        store.commit_history_cursor(mailbox.id, 500)
        # A renewal reports Gmail's current historyId, which must not clobber a
        # cursor the worker has already moved.
        store.activate_watch(mailbox.id, history_id=100, topic=TOPIC, expiration=utcnow() + timedelta(days=7))
        assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 500

    def test_commit_history_cursor_never_rewinds(self, store):
        mailbox = connect_mailbox(store, history_id=100)
        store.commit_history_cursor(mailbox.id, 900)
        store.commit_history_cursor(mailbox.id, 200)
        assert store.get_mailbox_by_db_id(mailbox.id).last_committed_history_id == 900

    def test_disconnect_clears_credential_and_frees_the_address(self, store):
        mailbox = connect_mailbox(store)
        store.disconnect_mailbox(mailbox.id)

        gone = store.get_mailbox_by_db_id(mailbox.id)
        assert gone.status == "disconnected"
        assert gone.refresh_token_ciphertext is None
        assert store.find_live_mailbox_by_address("user@example.com") is None

        # The same address can be connected again afterwards.
        reconnected = connect_mailbox(store, tenant="acme", mailbox="inbox-2")
        assert store.find_live_mailbox_by_address("user@example.com").id == reconnected.id

    def test_reconnect_same_address_updates_existing_live_mailbox(self, store):
        """A second connect for an already-live address refreshes the existing
        mailbox instead of raising a duplicate-key error (one live mailbox per
        address). Regression for the reconnect 500."""
        first = connect_mailbox(store, tenant="acme", mailbox="inbox")

        reconnected = store.upsert_mailbox(
            tenant_id="acme",
            mailbox_id="inbox-again",
            email_address="user@example.com",
            refresh_token_ciphertext=b"new-ciphertext",
            token_key_id="rotated-key",
        )

        assert reconnected.id == first.id  # same row, not a new one
        assert reconnected.refresh_token_ciphertext == b"new-ciphertext"
        assert reconnected.status == "pending"
        assert store.find_live_mailbox_by_address("user@example.com").id == first.id

    def test_watch_renewal_due_list_respects_expiry(self, store):
        soon = store.upsert_mailbox(
            tenant_id="acme",
            mailbox_id="soon",
            email_address="soon@example.com",
            refresh_token_ciphertext=b"x",
            token_key_id="k",
        )
        store.activate_watch(soon.id, history_id=1, topic=TOPIC, expiration=utcnow() + timedelta(hours=6))
        later = store.upsert_mailbox(
            tenant_id="acme",
            mailbox_id="later",
            email_address="later@example.com",
            refresh_token_ciphertext=b"x",
            token_key_id="k",
        )
        store.activate_watch(later.id, history_id=1, topic=TOPIC, expiration=utcnow() + timedelta(days=6))

        due = store.mailboxes_due_for_watch_renewal(before=utcnow() + timedelta(days=1))
        assert [m.mailbox_id for m in due] == ["soon"]

    # --- Notifications / jobs --------------------------------------------
    def test_valid_notification_creates_durable_job(self, store):
        mailbox = connect_mailbox(store)
        job = notify(store, "user@example.com", 150, "pubsub-1")

        assert job is not None
        assert job.status == "pending"
        assert job.requested_history_id == 150
        # Durable: readable back by ID, not just returned from the call.
        assert store.get_job(job.id).mailbox_db_id == mailbox.id

    def test_duplicate_pubsub_message_creates_no_duplicate_work(self, store):
        connect_mailbox(store)
        first = notify(store, "user@example.com", 150, "pubsub-dup")
        again = notify(store, "user@example.com", 150, "pubsub-dup")

        assert first is not None
        assert again is None  # redelivery: acked, no new job
        assert store.claim_job(owner="w1") is not None
        assert store.claim_job(owner="w1") is None  # exactly one job existed

    def test_higher_history_id_coalesces_using_numeric_order(self, store):
        connect_mailbox(store)
        first = notify(store, "user@example.com", 9, "pubsub-a")
        second = notify(store, "user@example.com", 10, "pubsub-b")

        # Text ordering would say '10' < '9' and keep 9. Numeric says 10.
        assert first.id == second.id
        assert store.get_job(first.id).requested_history_id == 10

        third = notify(store, "user@example.com", 99999, "pubsub-c")
        notify(store, "user@example.com", 100000, "pubsub-d")
        assert store.get_job(third.id).requested_history_id == 100000

    def test_lower_history_id_does_not_lower_a_pending_job(self, store):
        connect_mailbox(store)
        job = notify(store, "user@example.com", 500, "pubsub-high")
        notify(store, "user@example.com", 100, "pubsub-low")
        assert store.get_job(job.id).requested_history_id == 500

    def test_notification_for_unknown_address_is_acked_without_work(self, store):
        connect_mailbox(store)
        assert notify(store, "stranger@example.com", 150, "pubsub-x") is None
        assert store.claim_job(owner="w1") is None

    def test_notification_for_disconnected_mailbox_creates_no_job(self, store):
        mailbox = connect_mailbox(store)
        store.disconnect_mailbox(mailbox.id)
        assert notify(store, "user@example.com", 150, "pubsub-y") is None

    def test_claim_is_exclusive_and_increments_attempts(self, store):
        connect_mailbox(store)
        notify(store, "user@example.com", 150, "pubsub-1")

        claimed = store.claim_job(owner="worker-1")
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.attempts == 1
        assert claimed.lease_owner == "worker-1"
        # A second worker must not pick up a job that is already leased.
        assert store.claim_job(owner="worker-2") is None

    def test_failed_job_returns_to_pending_for_retry(self, store):
        connect_mailbox(store)
        notify(store, "user@example.com", 150, "pubsub-1")
        claimed = store.claim_job(owner="worker-1")

        store.fail_job(claimed.id, "boom")
        assert store.get_job(claimed.id).status == "pending"

        retried = store.claim_job(owner="worker-2")
        assert retried.id == claimed.id
        assert retried.attempts == 2

    def test_job_stops_retrying_after_max_attempts(self, store):
        connect_mailbox(store)
        notify(store, "user@example.com", 150, "pubsub-1")
        job = store.claim_job(owner="w")
        store.fail_job(job.id, "boom", max_attempts=1)

        assert store.get_job(job.id).status == "failed"
        assert store.claim_job(owner="w") is None

    def test_failure_folds_into_a_newer_pending_job(self, store):
        connect_mailbox(store)
        first = notify(store, "user@example.com", 150, "pubsub-1")
        store.claim_job(owner="worker-1")
        # A newer notification arrives while the first job is running.
        second = notify(store, "user@example.com", 300, "pubsub-2")
        assert second.id != first.id

        store.fail_job(first.id, "boom")
        # Only one pending job per mailbox, and it covers both history IDs.
        assert store.get_job(first.id).status == "failed"
        assert store.get_job(second.id).requested_history_id == 300

        claimed = store.claim_job(owner="worker-2")
        assert claimed.id == second.id
        assert store.claim_job(owner="worker-3") is None

    def test_completed_job_is_not_reclaimed(self, store):
        connect_mailbox(store)
        notify(store, "user@example.com", 150, "pubsub-1")
        job = store.claim_job(owner="w")
        store.complete_job(job.id)

        assert store.get_job(job.id).status == "done"
        assert store.claim_job(owner="w") is None

    def test_tenant_and_mailbox_isolation(self, store):
        acme = connect_mailbox(store, tenant="acme", mailbox="inbox", address="a@example.com")
        globex = connect_mailbox(store, tenant="globex", mailbox="inbox", address="b@example.com")

        acme_job = notify(store, "a@example.com", 111, "pubsub-acme")
        globex_job = notify(store, "b@example.com", 222, "pubsub-globex")

        assert acme_job.tenant_id == "acme" and acme_job.mailbox_db_id == acme.id
        assert globex_job.tenant_id == "globex" and globex_job.mailbox_db_id == globex.id

        # A cursor commit for one tenant must not touch the other's.
        store.commit_history_cursor(acme.id, 999)
        assert store.get_mailbox_by_db_id(globex.id).last_committed_history_id == 100

    def test_stored_mailbox_never_exposes_a_plaintext_token(self, store):
        mailbox = connect_mailbox(store)
        reloaded = store.get_mailbox_by_db_id(mailbox.id)
        # The store round-trips ciphertext bytes and nothing else; the repr is
        # what ends up in log lines and pytest failure output.
        assert reloaded.refresh_token_ciphertext == b"ciphertext-not-a-token"
        assert "ciphertext" not in repr(reloaded)
        assert "refresh_token" not in repr(reloaded)


@pytest.fixture
def store():  # pragma: no cover - overridden by each subclass
    raise NotImplementedError("provide a store fixture")
