import logging
from fastapi import APIRouter, Request, Depends

from utils.sessions import ChatSession, get_chat_session
from routes.librarian_router import search_books_api  
from routes.library_info_router import library_info
from utils.text_utils import fuzzy_match_keywords
from utils.language_translator import detect_language, translate_to_english

logger = logging.getLogger("query_router")
router = APIRouter()

# Unified keyword pool (route logic now handled in search_books_api)
SEARCH_BOOKS_KEYWORDS = {
    "search", "quantity", "title", "find", "lookup", "copies", "available",
    "recommend", "suggest", "recommendation", "like", "similar", "same",
    "isbn"
}

@router.post("/query_router")
async def query_router(
    request: Request,
    chat_session: ChatSession = Depends(get_chat_session)
):
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        logger.info(f"Received query: '{user_query}'")

        if not user_query:
            logger.warning("No query parameter provided.")
            return {"error": "Query parameter is required"}

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)
        logger.info(f"Translated query: '{translated_query}'")

        tokens = translated_query.lower().split()
        logger.info(f"Tokens (EN): {tokens}")

        if fuzzy_match_keywords(tokens, SEARCH_BOOKS_KEYWORDS, threshold=75):
            logger.info("Routing to: search_books_api")
            return await search_books_api(request, chat_session)

        logger.info("Routing to: library_info (fallback)")
        return await library_info(request, chat_session)

    except Exception as e:
        logger.error(f"Error in query_router: {e}", exc_info=True)
        return {"error": "Internal server error."}
