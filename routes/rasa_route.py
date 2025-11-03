import httpx
from fastapi import APIRouter, Depends
from utils.sessions import ChatSession, get_chat_session
from pydantic import BaseModel
import logging

router = APIRouter()
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"
logger = logging.getLogger("rasa_route")

class RasaRequest(BaseModel):
    query: str

@router.post("/ask_librarian")
async def ask_librarian(payload: RasaRequest, chat_session: ChatSession = Depends(get_chat_session)):
    async with httpx.AsyncClient() as client:
        rasa_resp = await client.post(
            RASA_URL,
            json={"sender": chat_session.session_id, "message": payload.query},
            timeout=60.0
        )

    messages = rasa_resp.json()
    logger.info(f"Raw Rasa response: {messages}")

    replies, books = [], []

    for m in messages:
        # Handle plain text responses (from general_query or other actions)
        if "text" in m:
            replies.append(m["text"])

        # Handle custom payloads
        if "custom" in m:
            custom = m["custom"]

            # Case 1: single dict
            if isinstance(custom, dict):
                if custom.get("type") == "book":
                    books.append(custom)
                elif custom.get("type") == "text":
                    replies.append(custom.get("answer", ""))

            # Case 2: list of dicts
            elif isinstance(custom, list):
                for item in custom:
                    if isinstance(item, dict):
                        if item.get("type") == "book":
                            books.append(item)
                        elif item.get("type") == "text":
                            replies.append(item.get("answer", ""))

    # If books found, return structured response
    if books:
        return {
            "response": [
                {
                    "type": "booksearch",
                    "answer": " ".join(replies) if replies else "Here are some books I found:",
                    "books": books
                }
            ]
        }

    # If text responses found, return them
    if replies:
        return {
                "answer": " ".join(replies),
        }

    # Final fallback to LLM
    from utils.llm_client import generate_response
    llm_answer = await generate_response(payload.query)
    logger.info(f"Fallback LLM response: {llm_answer}")
    return {
                "answer": llm_answer,
    }
