import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from routes.librarian_router import search_books_logic
from routes.library_info_router_test import library_info_logic
from utils.text_utils import fuzzy_match_keywords
from motor.motor_asyncio import AsyncIOMotorDatabase
# from routes.library_info_router import library_info
from db.connection import get_db
from routes.librarian_router import search_books_logic
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
     db: AsyncIOMotorDatabase = Depends(get_db), # <<< ADDED
 ):
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        logger.info(f"Received query: '{user_query}'")

        if not user_query:
            logger.warning("No query parameter provided.")
            return {"error": "Query parameter is required"}
        
        if not all(k in data for k in ["cardnumber", "sessionId"]):
             logger.error(f"Incomplete request to query_router. Missing cardnumber or sessionId. Data: {data}")
             return JSONResponse(
                 content={"error": "Request body must include 'cardnumber' and 'sessionId'."}, 
                 status_code=400
            )

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)
        logger.info(f"Translated query: '{translated_query}'")

        tokens = translated_query.lower().split()
        logger.info(f"Tokens (EN): {tokens}")

        if fuzzy_match_keywords(tokens, SEARCH_BOOKS_KEYWORDS, threshold=75):
            logger.info("Routing to: search_books_api")
            logger.info(f"Routing to search_books_logic with data: {data}")
            return await search_books_logic(data, db)

        logger.info("Routing to: library_info (fallback) with data: {data}")
        return await library_info_logic(data, db)

    except Exception as e:
        logger.error(f"Error in query_router: {e}", exc_info=True)
        return {"error": "Internal server error."}