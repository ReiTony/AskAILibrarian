import logging
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse 
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.sessions import get_session_and_user_data
from routes.librarian_router import search_books_api  
from routes.library_info_router import library_info
from utils.prompt_templates import get_router_prompt
from utils.llm_client import generate_response
from utils.language_translator import detect_language, translate_to_english
from db.connection import get_db

logger = logging.getLogger("query_router")
router = APIRouter()


@router.post("/query_router")
async def query_router(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "").strip()
        logger.info(f"Routing query for cardnumber '{cardnumber}': '{user_query} : {chat_session.session_id}'")

        if not user_query:
            logger.warning("No query parameter provided.")
            # Return a proper JSONResponse object
            return JSONResponse(
                content={"error": "Query parameter is required"}, 
                status_code=400
            )

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)
        logger.info(f"Translated query: '{translated_query}'")

        tokens = translated_query.lower().split()
        logger.info(f"Tokens (EN): {tokens}")

        recent_history = await chat_session.get_history() 
        history_text = "\n".join(f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in recent_history)
        router_prompt = get_router_prompt(history_text, user_query)

        logger.info(f"=====Router Prompt for LLM=====\n{router_prompt}\n==========================")
        
        # --- ROUTING LOGIC ---
        # Decide which function to call
        try:
            route_decision = (await generate_response(router_prompt)).strip().lower()
            logger.info(f"LLM Route Decision: '{route_decision}'")
        except Exception as e:
            logger.error(f"LLM router failed: {e}. Defaulting to general_information.")
            route_decision = "general_information"
            
        # 4. Use the decision to select the target function
        if "book_search" in route_decision:
            logger.info("Routing to: search_books_api")
            target_function = search_books_api
        else:
            logger.info("Routing to: library_info (fallback)")
            target_function = library_info

        # --- EXECUTION AND ERROR HANDLING ---
        # Now, call the chosen function and handle its potential errors
        try:
            # The 'await' call is now inside its own focused try block
            response = await target_function(session_data, db)
            return response
        except Exception as e:
            # This will catch any UNHANDLED exception from the downstream router
            # e.g., if search_books_api itself crashed before it could return a JSONResponse
            logger.error(
                f"An unhandled exception occurred in the downstream router '{target_function.__name__}': {e}", 
                exc_info=True
            )
            # Return a generic 500 error to the client
            return JSONResponse(
                content={"error": f"An internal error occurred while processing the request in {target_function.__name__}."},
                status_code=500
            )

    except Exception as e:
        # This outer block catches errors in the query_router itself 
        # (e.g., failed request.json(), translation error)
        logger.error(f"Critical error in query_router itself: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "A critical internal error occurred in the main router."},
            status_code=500
        )