from fastapi import Header, Depends
from uuid import uuid4
from collections import defaultdict, deque
from typing import Optional

# In-memory bubble: session_id -> recent messages (max 10)
MAX_HISTORY_LENGTH = 10
_memory_bubble = defaultdict(lambda: deque(maxlen=MAX_HISTORY_LENGTH))

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def get_history(self) -> list:
        """Retrieve chat messages in chronological order."""
        return list(_memory_bubble[self.session_id])

    async def add_message(self, role: str, content: str):
        """Add a message to the memory bubble."""
        _memory_bubble[self.session_id].append({"role": role, "content": content})

# Extract/generate session id
async def get_session_id(session_id: Optional[str] = Header(None)) -> str:
    return session_id or str(uuid4())

# Inject a ChatSession object
async def get_chat_session(
    session_id: str = Depends(get_session_id)
) -> ChatSession:
    return ChatSession(session_id=session_id)
