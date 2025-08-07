import logging
from fastapi import Header, Depends, Request
from uuid import uuid4
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("session_manager")

# ---------------------------------------
# In-memory chat history (per session)
# ---------------------------------------
MAX_HISTORY_LENGTH = 10
_memory_bubble = defaultdict(lambda: deque(maxlen=MAX_HISTORY_LENGTH))


class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.cardnumber: Optional[str] = None  # Track user identity across requests

    async def get_history(self) -> list[dict]:
        """Retrieve session's message history (FIFO)."""
        return list(_memory_bubble[self.session_id])

    async def add_message(self, role: str, content: str) -> None:
        """Append a user or assistant message to session memory."""
        _memory_bubble[self.session_id].append({"role": role, "content": content})


# ----------------------------
# Dependency: Session ID
# ----------------------------
async def get_session_id(session_id: Optional[str] = Header(None)) -> str:
    """Get session ID from headers or generate a new one."""
    return session_id or str(uuid4())


# ----------------------------
# Dependency: ChatSession
# ----------------------------
async def get_chat_session(
    session_id: str = Depends(get_session_id)
) -> ChatSession:
    """Get or create a ChatSession object tied to session ID."""
    return ChatSession(session_id=session_id)


# -----------------------------------------------
# ✅ Unified Dependency: Session + Cardnumber + Data
# -----------------------------------------------
async def get_session_and_user_data(
    request: Request,
    chat_session: ChatSession = Depends(get_chat_session)
) -> Tuple[ChatSession, str, Dict[str, Any]]:
    """
    Fetches:
    - Chat session (in-memory)
    - Persisted cardNumber (user ID)
    - Parsed request body

    Use this as the main dependency for all librarian routes.
    """
    # Attempt to parse JSON body
    request_data: Dict[str, Any] = {}
    try:
        request_data = await request.json()
    except Exception:
        logger.debug("[Session Manager] No JSON body or invalid format.")

    # Retrieve or attach cardNumber to session
    cardnumber = getattr(chat_session, "cardNumber", None)
    if not cardnumber:
        cardnumber = request_data.get("cardNumber")
        if cardnumber:
            chat_session.cardNumber = cardnumber
            logger.info(f"[Session Manager] CardNumber '{cardnumber}' attached to session {chat_session.session_id}.")
        else:
            logger.warning(f"[Session Manager] Missing cardNumber for session {chat_session.session_id}.")

    return chat_session, cardnumber, request_data
