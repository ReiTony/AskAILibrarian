import logging
from fastapi import APIRouter, Request, Depends, HTTPException

from utils.sessions import ChatSession, get_chat_session
from routes.search_books_route import search_books_api
from routes.recommend_books_route import recommend_books
from routes.library_info_routes import library_info
from routes.lookup_book_route import lookup_isbn  
from utils.text_utils import fuzzy_match_keywords
from utils.language_translator import detect_language, translate_to_english

logger = logging.getLogger("query_router")
router = APIRouter()

SEARCH_BOOKS_KEYWORDS = {"search", "quantity", "title", "find", "lookup", "copies", "available"}
RECOMMEND_BOOKS_KEYWORDS = {"recommend", "suggest", "recommendation", "like", "similar", "same"}
ISBN_LOOKUP_KEYWORDS = {"isbn"}  

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
        logger.info(f"Translated query for routing: '{translated_query}'")

        tokens = translated_query.lower().split()
        logger.info(f"Tokens (EN): {tokens}")
        
        if fuzzy_match_keywords(tokens, ISBN_LOOKUP_KEYWORDS):
            logger.info("Routing to: lookup_isbn")
            return await lookup_isbn(request, chat_session)

        if fuzzy_match_keywords(tokens, SEARCH_BOOKS_KEYWORDS):
            logger.info("Routing to: search_books_api")
            return await search_books_api(request, chat_session)

        if fuzzy_match_keywords(tokens, RECOMMEND_BOOKS_KEYWORDS):
            logger.info("Routing to: recommend_books")
            return await recommend_books(request, chat_session)

        logger.info("Routing to: library_info (fallback handler)")
        return await library_info(request, chat_session)

    except Exception as e:
        logger.error(f"Error in query_router: {e}", exc_info=True)
        return {"error": "Internal server error."}
