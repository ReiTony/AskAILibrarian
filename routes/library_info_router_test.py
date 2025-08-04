import logging
import re

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.connection import get_db
from utils.chat_retention import save_message_and_get_context
from utils.chroma_client import web_db
from utils.suggestions import get_suggestions, default_reminders
from utils.llm_client import generate_response
from utils.prompt_templates import library_fallback_prompt, library_contextual_prompt
from utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language,
)

router = APIRouter()
logger = logging.getLogger("library_info_route")

# ... (LOCATION_ALIASES, STOPWORDS, simplify_query, detect_locations, format_response are all unchanged) ...
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


# <<< --- NEW: Decoupled logic function --- >>>
async def library_info_logic(data: dict, db: AsyncIOMotorDatabase):
    try:
        user_query = data.get("query", "")
        # --- ADDED: Get state management keys from data ---
        cardnumber = data.get("cardnumber")
        session_id = data.get("sessionId")

        if not all([user_query, cardnumber, session_id]):
            return JSONResponse(
                content={"error": "Logic error: 'query', 'cardnumber', and 'sessionId' are required."},
                status_code=400,
            )

        # --- MODIFIED: Use new chat retention utility ---
        # Save user message and get context from DB in one step
        history = await save_message_and_get_context(
            db, cardnumber, session_id, "user", user_query
        )
        history_text = "\n".join(
            f"{'Human' if msg['sender']=='user' else 'AI'}: {msg['text']}"
            for msg in history
        )

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)

        # ... (The entire intent detection, Chroma search, and response generation logic is the same) ...
        # ... (I am including it here for completeness) ...
        intent_prompt = (
            "Classify the user's intent.\n\n"
            "If the question is related to library services (like books, borrowing, locations, library staff, etc.), reply with: library\n"
            "If the question is general, personal, or off-topic (like greetings, languages, preferences, abilities), reply with: general\n\n"
            f"User Question: {translated_query}\n"
            "Intent:"
        )
        intent = "library" # Default
        try:
            intent_response = await generate_response(intent_prompt)
            intent = intent_response.strip().lower()
            logger.info(f"[Intent Detection] Query: '{translated_query}' → Intent: {intent}")
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")

        bot_response = "Sorry, I am unable to answer that right now." # Default response

        if intent == "general":
            prompt = library_fallback_prompt(history_text, translated_query)
            bot_response = await generate_response(prompt)
            suggestions = default_reminders
        else: # library intent
            try:
                simple_query = simplify_query(translated_query)
                results_original = web_db.similarity_search_with_score(translated_query, k=5)
                results_simple = web_db.similarity_search_with_score(simple_query, k=5) if simple_query != translated_query else []
                best_results = results_original
                if results_simple:
                    top_score_original = max([score for _, score in results_original], default=0)
                    top_score_simple = max([score for _, score in results_simple], default=0)
                    if top_score_simple > top_score_original:
                        best_results = results_simple
                results = best_results
            except Exception as e:
                logger.error(f"Chroma similarity search error: {e}")
                results = []

            query_keywords = [word.lower() for word in translated_query.split() if len(word) > 3]
            matching_locations = detect_locations(translated_query)
            prioritized_results = []
            for doc, score in results:
                content = doc.page_content.lower()
                match_count = sum(keyword in content for keyword in query_keywords)
                prioritized_results.append((doc, match_count, score))

            try:
                suggestions = get_suggestions(translated_query, query_keywords)
            except Exception as e:
                logger.error(f"Suggestion generation error: {e}")
                suggestions = default_reminders

            try:
                if len(matching_locations) > 1:
                    clarification_msg = (
                        f"Your question could refer to multiple locations. "
                        f"Did you mean: {', '.join(matching_locations)}? "
                        "Please specify which library or division you are asking about."
                    )
                    bot_response = clarification_msg

                elif len(matching_locations) == 1:
                    location = matching_locations[0].lower()
                    filtered = [
                        (doc, mc, sc)
                        for (doc, mc, sc) in prioritized_results
                        if location in doc.page_content.lower()
                    ]
                    prioritized_results = filtered or prioritized_results
                    prioritized_results.sort(key=lambda x: (-x[1], x[2]))
                    if prioritized_results:
                        top_doc = prioritized_results[0][0]
                        context = f"{top_doc.metadata.get('main_section', 'General')}: {top_doc.page_content}"
                        prompt = library_contextual_prompt(context, history_text, translated_query)
                        try:
                            bot_response = await generate_response(prompt)
                        except Exception as e:
                            logger.error(f"LLM response error: {e}")
                            bot_response = "Sorry, I could not generate an answer right now."
                    else:
                        prompt = library_fallback_prompt(history_text, translated_query)
                        try:
                            bot_response = await generate_response(prompt)
                        except Exception as e:
                            logger.error(f"LLM fallback error: {e}")
                            bot_response = "Sorry, I could not generate an answer right now."
                else:
                    filtered = [
                        (doc, mc, sc)
                        for (doc, mc, sc) in prioritized_results
                        if "university library" in doc.page_content.lower()
                    ]
                    prioritized_results = filtered or prioritized_results
                    prioritized_results.sort(key=lambda x: (-x[1], x[2]))
                    if prioritized_results:
                        top_doc = prioritized_results[0][0]
                        context = f"{top_doc.metadata.get('main_section', 'General')}: {top_doc.page_content}"
                        prompt = library_contextual_prompt(context, history_text, translated_query)
                        try:
                            bot_response = await generate_response(prompt)
                        except Exception as e:
                            logger.error(f"LLM response error: {e}")
                            bot_response = "Sorry, I could not generate an answer right now."
                    else:
                        prompt = library_fallback_prompt(history_text, translated_query)
                        try:
                            bot_response = await generate_response(prompt)
                        except Exception as e:
                            logger.error(f"LLM fallback error: {e}")
                            bot_response = "Sorry, I could not generate an answer right now."

            except Exception as e:
                logger.error(f"Response formatting error: {e}")
                return JSONResponse(
                    content={"error": "Internal server error."},
                    status_code=500
                )
            
        translated_response = await translate_to_user_language(bot_response, user_lang)
        response_content = format_response(translated_response, suggestions)

        # --- MODIFIED: Save assistant response to the DB ---
        await save_message_and_get_context(
            db, cardnumber, session_id, "assistant", translated_response
        )

        # --- MODIFIED: Fetch final history from DB to return to client ---
        final_history = await save_message_and_get_context(db, cardnumber, session_id, "assistant", "") # Empty message just retrieves
        response_content["history"] = final_history

        return JSONResponse(content=response_content, status_code=200)

    except Exception as e:
        logger.error(f"Library info logic failed: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "Internal server error."},
            status_code=500
        )

# <<< --- NEW: Thin API endpoint wrapper --- >>>
@router.post("/library_info")
async def library_info_api(request: Request, db: AsyncIOMotorDatabase = Depends(get_db)):
    """Endpoint to handle library info queries. It reads the request and passes it to the logic function."""
    data = await request.json()
    return await library_info_logic(data, db)