import logging
import re
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.sessions import get_session_and_user_data
from utils.chroma_client import web_db
from utils.suggestions import get_suggestions, default_reminders
from utils.llm_client import generate_response
from utils.prompt_templates import library_fallback_prompt
from utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language,
)
from utils.chat_retention import get_retained_history, save_conversation_turn
from db.connection import get_db

router = APIRouter()
logger = logging.getLogger("library_info_route")

# -------------------- Config --------------------
LOCATION_ALIASES = {
    "University Library": ["university library", "main library", "msu library", "campus library"],
    "American Corner Marawi": ["american corner", "american library", "ac marawi"],
    "Access Services Division": ["access services", "access division"],
    "Administrative Services Division": ["admin services", "administrative division"],
    "Malano UPHub": ["malano uphub", "uphup"],
}

STOPWORDS = {
    "what", "are", "the", "is", "at", "to", "a", "of", "for", "i", "do", "in", "on", "by", "can",
    "how", "and", "an", "does", "with", "from", "my", "me", "about", "obtain", "get", "apply",
    "steps", "process", "please", "provide", "information", "explain", "give", "list", "details",
    "tell", "need", "show", "am", "required", "requirements", "card", "way", "would", "like",
}

# ------------------ Utilities ------------------
def simplify_query(query: str) -> str:
    q = re.sub(r"[^\w\s]", "", query.lower())
    return " ".join(word for word in q.split() if word not in STOPWORDS)

def detect_locations(query: str) -> list[str]:
    matches = []
    query_lower = query.lower()
    for loc, aliases in LOCATION_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            matches.append(loc)
    return matches

def format_response(answer: str, suggestions: list) -> dict:
    response = {"answer": answer.strip()}
    prefix = "reminder" if suggestions == default_reminders else "suggestion"
    for i, suggestion in enumerate(suggestions[:3], 1):
        response[f"{prefix}{i}"] = suggestion
    return response

# ------------------ Main Route ------------------
@router.post("/library_info")
async def library_info(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db),
    intent: str = None
):
    try:
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "").strip()
        cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None)
        if not user_query:
            return JSONResponse(content={"error": "Query is required."}, status_code=422)

        logger.info(f"[library_info] Query received from cardnumber={cardnumber}")

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)

        # Build LLM history context
        retained = await get_retained_history(db, cardnumber)
        recent = await chat_session.get_history()
        history = retained + recent[-4:]
        history_text = "\n".join(
            f"{'Human' if h['role'] == 'user' else 'AI'}: {h['content']}"
            for h in history
        )

        # Default suggestions + fallback response
        suggestions = default_reminders
        fallback_prompt = library_fallback_prompt(history_text, translated_query)

        # ----- Generate Response -----
        try:
            simple_query = simplify_query(translated_query)
            # Try searching ChromaDB for relevant info (e.g., locations, policies)
            results = web_db.similarity_search(simple_query or translated_query, k=3)

            if results:
                context = "\n---\n".join(doc.page_content for doc in results)
                prompt = library_fallback_prompt(history_text + "\n\n" + context, translated_query)
            else:
                prompt = fallback_prompt

            response_text = await generate_response(prompt)
            suggestions = get_suggestions(translated_query, [])
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            response_text = "Sorry, I encountered an error while processing your request."

        # ----- Save + Respond -----
        translated_response = await translate_to_user_language(response_text, user_lang)

        # Save history (short + long)
        await chat_session.add_message("user", user_query)
        await chat_session.add_message("assistant", translated_response)
        await save_conversation_turn(db, cardnumber, user_query, translated_response)

        logger.info(f"[Chat Saved] Successfully saved turn for cardnumber={cardnumber}")

        # Build and return response
        response_payload = format_response(translated_response, suggestions)
        response_payload["history"] = await chat_session.get_history()
        return JSONResponse(content=response_payload, status_code=200)

    except Exception as e:
        logger.error(f"[Fatal] /library_info failed: {e}", exc_info=True)
        return JSONResponse(content={"error": "Internal server error."}, status_code=500)