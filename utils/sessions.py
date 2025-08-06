import logging
from fastapi import Header, Depends, Request
from uuid import uuid4
from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Any

# Get a logger instance
logger = logging.getLogger("session_manager")

# In-memory bubble: session_id -> recent messages (max 10)
MAX_HISTORY_LENGTH = 10
_memory_bubble = defaultdict(lambda: deque(maxlen=MAX_HISTORY_LENGTH))

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # We add an attribute to hold the cardnumber, it starts as None
        self.cardnumber: Optional[str] = None

    async def get_history(self) -> list:
        """Retrieve chat messages in chronological order."""
        return list(_memory_bubble[self.session_id])

    async def add_message(self, role: str, content: str):
        """Add a message to the memory bubble."""
        _memory_bubble[self.session_id].append({"role": role, "content": content})

# Extract/generate session id (No changes)
async def get_session_id(session_id: Optional[str] = Header(None)) -> str:
    return session_id or str(uuid4())

# Inject a ChatSession object (No changes)
async def get_chat_session(
    session_id: str = Depends(get_session_id)
) -> ChatSession:
    # This now returns the class that can hold a cardnumber
    return ChatSession(session_id=session_id)

# =====================================================================
# NEW STATEFUL DEPENDENCY - THE COMPLETE SOLUTION
# =====================================================================
async def get_session_and_user_data(
    request: Request, 
    chat_session: ChatSession = Depends(get_chat_session)
) -> Tuple[ChatSession, str, Dict[str, Any]]:
    """
    A stateful dependency that gets the session, persists the user's cardnumber
    across requests, and provides the request body.

    This should be the ONLY session-related dependency your routes use.

    Returns:
        A tuple containing: (chat_session, cardnumber, request_data)
    """
    # 1. Try to read the JSON body once and store it.
    request_data = {}
    try:
        request_data = await request.json()
    except Exception:
        # Handle cases with no body or non-JSON body gracefully
        pass

    # 2. Try to get cardnumber from the session object first (from a previous request)
    cardnumber = getattr(chat_session, 'cardnumber', None)

    # 3. If not on the session, check the incoming request body
    if not cardnumber:
        cardnumber_from_request = request_data.get("cardnumber")
        if cardnumber_from_request:
            # If found in the request, ATTACH IT to the session object for future requests.
            chat_session.cardnumber = cardnumber_from_request
            cardnumber = cardnumber_from_request
            logger.info(f"Cardnumber '{cardnumber}' found in request and attached to session {chat_session.session_id}.")
            
    if not cardnumber:
        logger.warning(f"Cardnumber not found for session {chat_session.session_id}. History will be incomplete.")

    # 4. Return everything the route needs in one go
    return chat_session, cardnumber, request_data