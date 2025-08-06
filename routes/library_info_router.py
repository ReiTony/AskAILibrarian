import logging
import re
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from utils.sessions import get_session_and_user_data    
from utils.chroma_client import web_db
from utils.suggestions import get_suggestions, default_reminders
from utils.llm_client import generate_response
from utils.prompt_templates import library_fallback_prompt, get_intent_prompt_few_shot
from utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language,
)
from utils.chat_retention import get_retained_history, save_conversation_turn
from db.connection import get_db

router = APIRouter()
logger = logging.getLogger("library_info_route")

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

# --- Utility Functions (No Changes) ---
def simplify_query(query):
    q = query.lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = " ".join(q.split())
    words = [word for word in q.split() if word not in STOPWORDS]
    return " ".join(words)

def detect_locations(query):
    matches = []
    query_lower = query.lower()
    for loc, aliases in LOCATION_ALIASES.items():
        for alias in aliases:
            if alias in query_lower:
                matches.append(loc)
                break
    return matches

def format_response(answer: str, suggestions: list) -> dict:
    response = {"answer": answer.strip()}
    prefix = "reminder" if suggestions == default_reminders else "suggestion"
    for i, suggestion in enumerate(suggestions[:3], 1):
        response[f"{prefix}{i}"] = suggestion
    return response

# --- Main API Route (Refactored) ---
@router.post("/library_info")
async def library_info(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        # --- 1. SETUP AND DATA GATHERING ---
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "")
        cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None) #Huwag alisin. For CardNumber compatibility

        if not isinstance(user_query, str) or not user_query.strip():
            return JSONResponse(content={"error": "Query is required."}, status_code=422)
        
        logger.info(f"Processing library_info for cardnumber: '{cardnumber}'")

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)
        
        retained_history = await get_retained_history(db, cardnumber)   
        session_history = await chat_session.get_history()
        history_for_router = session_history[-4:]
        history_text = "\n".join(f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in (retained_history + history_for_router))

        logger.info(f"History for '{cardnumber}' built. Total messages for context: {len(history_text)}")

        # --- 2. RESPONSE GENERATION ---
        # Initialize variables that will be populated by the logic below
        bot_response = ""
        suggestions = default_reminders

        contextPrompt = library_fallback_prompt(history_text, translated_query)

        # Intent detection
        try:
            intent_response = get_intent_prompt_few_shot(history_text, translated_query)
            llmResponse = await generate_response(intent_response)
            logger.info(f"\n\n[Intent Detection] Raw intent response: {intent_response}\n\n")
            intent = llmResponse.strip().lower()
            logger.info(f"[Intent Detection] Query: '{translated_query}' → Intent: {intent}")
        except Exception as e:
            logger.warning(f"Intent detection failed, defaulting to 'library': {e}")
            intent = "library"

        if intent == "general":
            prompt = contextPrompt
            logger.info(f"\n======Prompt for general intent:======== {prompt}\n\n")
            bot_response = await generate_response(prompt)
            # 'suggestions' will remain as the default_reminders

        else:  # This block handles intent == "library" or any fallback
            try:
                simple_query = simplify_query(translated_query)
                results = web_db.similarity_search(simple_query or translated_query, k=3)
                
                if not results:
                    prompt = contextPrompt
                    logger.info(f"\n======Prompt for general intent:======== {prompt}\n\n")
                else:
                    context = "\n---\n".join([doc.page_content for doc in results])
                    prompt = contextPrompt
                    logger.info(f"\n======Prompt for general intent:======== {prompt}\n\n")
                
                bot_response = await generate_response(prompt)
                suggestions = get_suggestions(translated_query, []) # Get fresh suggestions
            except Exception as e:
                logger.error(f"Error during library response generation: {e}", exc_info=True)
                bot_response = "Sorry, I encountered an error while looking for that information."

        # --- 3. FINALIZATION, SAVING, AND RETURN (CONVERGENCE POINT) ---

        # Translate the final AI response back to the user's language
        translated_response = await translate_to_user_language(bot_response, user_lang)

        # Save the full conversation turn to ALL storage locations
        try:
            # Save to short-term session memory
            await chat_session.add_message("user", user_query)
            await chat_session.add_message("assistant", translated_response)
            
            # Save to long-term retention database
            await save_conversation_turn(db, cardnumber, user_query, translated_response)
            logger.info(f"[Chat Retention] Saved library_info turn for {cardnumber}")
        except Exception as e:
            logger.error(f"Failed to save chat history for {cardnumber}: {e}")

        # Format the final JSON response for the frontend
        response_content = format_response(translated_response, suggestions)
        response_content["history"] = await chat_session.get_history()
        
        return JSONResponse(content=response_content, status_code=200)

    except Exception as e:
        logger.error(f"Library info endpoint failed catastrophically: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "A critical internal server error occurred."},
            status_code=500
        )