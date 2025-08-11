import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse 
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.sessions import get_session_and_user_data
from utils.intent_classifier import classify_intent
from utils.chat_retention import get_retained_history
from db.connection import get_db

# Handler imports
from routes.librarian_route import search_books_api  
from routes.library_info_route import library_info
from utils.general_info_handler import handle_general_info

logger = logging.getLogger("query_router")
router = APIRouter()

# Central mapping: intent -> handler
INTENT_DISPATCH = {
    "general_info": handle_general_info,
    "library_info": library_info,
    "book_search": search_books_api,
    "book_recommend": search_books_api,
    "book_lookup_isbn": search_books_api,
}

@router.post("/query_router")
async def query_router(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "").strip()
        cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None)
        logger.info(f"Routing query for cardnumber '{cardnumber}': '{user_query} : {chat_session.session_id}'")

        if not user_query:
            logger.warning("No query parameter provided.")
            return JSONResponse(
                content={"error": "Query parameter is required"}, 
                status_code=400
            )

        # Build chat history for LLM context (if needed by intent_classifier or handler)
        retained_history = await get_retained_history(db, cardnumber)
        recent_history = await chat_session.get_history()
        full_history = retained_history + recent_history
        history_text = "\n".join(
            f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            for msg in full_history[-4:]
        )

        # --- Central Intent Classification ---
        try:
            intent = await classify_intent(user_query, history_text)
            logger.info(f"Detected intent: '{intent}'")
        except Exception as e:
            logger.error(f"Intent classifier failed: {e}. Defaulting to general_info.")
            intent = "general_info"

        # --- Intent Dispatch ---
        handler = INTENT_DISPATCH.get(intent)
        if not handler:
            logger.error(f"No handler found for intent: {intent}")
            return JSONResponse(content={"error": f"No handler found for intent: {intent}"}, status_code=500)
        
        try:
            # Pass intent to handler for downstream logic (optional)
            response = await handler(session_data, db, intent=intent)
            return response
        except Exception as e:
            logger.error(
                f"Error in handler for intent '{intent}': {e}", exc_info=True
            )
            return JSONResponse(
                content={"error": f"Internal error in handler for intent: {intent}"},
                status_code=500
            )

    except Exception as e:
        logger.error(f"Critical error in query_router itself: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "A critical internal error occurred in the main router."},
            status_code=500
        )
