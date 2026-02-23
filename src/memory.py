"""
memory.py - JSON-based chat session persistence.

Stores conversation turns (user + assistant messages) to disk so that
history survives between process restarts.  The last N turns are injected
into the generation prompt to give the LLM conversational context.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

MAX_STORED_TURNS = 200

@dataclass
class Turn:
    """A single conversational exchange."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Session:
    """Full session state persisted to disk."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------

class ChatSession:
    """
    Manages conversation history with JSON persistence.

    Args:
        session_id:    Unique name for this session (used as the filename stem).
        storage_dir:   Directory where session JSON files are written.
        history_turns: Number of most-recent turns included in the prompt context.
    """

    def __init__(
        self,
        session_id: str = "default",
        storage_dir: str = "./sessions",
        history_turns: int = 3,
    ) -> None:
        self._history_turns = history_turns
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._session_file = self._storage_dir / f"{session_id}.json"
        self._session = self._load_or_create(session_id)
        logger.info(
            "ChatSession '%s' loaded (%d existing turns).",
            session_id,
            len(self._session.turns),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Append a new turn and persist immediately."""
        if role not in {"user", "assistant"}:
            raise ValueError(f"role must be 'user' or 'assistant', got '{role}'.")
        turn = Turn(role=role, content=content)
        self._session.turns.append(turn)

        if len(self._session.turns) > MAX_STORED_TURNS:
            self._session.turns = self._session.turns[-MAX_STORED_TURNS:]
            
        self._save()

    def get_history_for_prompt(self) -> list[dict[str, str]]:
        """
        Return the last *history_turns* full exchanges as a list of dicts
        compatible with the HF chat-completion message format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, …]

        We take the last (history_turns * 2) individual turns to capture
        complete user+assistant pairs.
        """
        recent_turns = self._session.turns[-(self._history_turns * 2):]
        return [{"role": t.role, "content": t.content} for t in recent_turns]

    def clear(self) -> None:
        """Delete all turns for the current session and persist."""
        logger.info("Clearing session '%s'.", self._session.session_id)
        self._session.turns = []
        self._save()

    @property
    def session_id(self) -> str:
        return self._session.session_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_create(self, session_id: str) -> Session:
        if self._session_file.exists():
            try:
                raw = json.loads(self._session_file.read_text(encoding="utf-8"))
                turns = [Turn(**t) for t in raw.get("turns", [])]
                return Session(session_id=raw["session_id"], turns=turns)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to load session from '%s': %s – starting fresh.",
                    self._session_file,
                    exc,
                )
        return Session(session_id=session_id)

    def _save(self) -> None:
        data = {
            "session_id": self._session.session_id,
            "turns": [asdict(t) for t in self._session.turns],
        }
        try:
            self._session_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist session to '%s': %s", self._session_file, exc)