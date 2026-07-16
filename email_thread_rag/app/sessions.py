from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from email_thread_rag.app.schemas import MemorySlots, SessionState, Turn
from email_thread_rag.config import Settings


class SessionStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._sessions: dict[str, SessionState] = {}

    def start_session(self, thread_id: str) -> SessionState:
        now = datetime.now(timezone.utc)
        session = SessionState(
            session_id=str(uuid4()),
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
            memory_slots=MemorySlots(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            return self._sessions[session_id].model_copy(deep=True)

    def save(self, session: SessionState) -> SessionState:
        session.updated_at = datetime.now(timezone.utc)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def switch_thread(self, session_id: str, thread_id: str) -> SessionState:
        session = self.get(session_id)
        session.thread_id = thread_id
        session.recent_turns.clear()
        session.memory_slots = MemorySlots()
        return self.save(session)

    def reset(self, session_id: str) -> SessionState:
        session = self.get(session_id)
        session.recent_turns.clear()
        session.memory_slots = MemorySlots()
        return self.save(session)

    def append_turn(self, session_id: str, role: str, text: str) -> SessionState:
        session = self.get(session_id)
        session.recent_turns.append(Turn(role=role, text=text, timestamp=datetime.now(timezone.utc)))
        session.recent_turns = session.recent_turns[-self.settings.max_recent_turns :]
        return self.save(session)
