import logging
import re

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from utils.sessions import ChatSession, get_chat_session
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

@router.post("/library_info")
async def library_info(
    request: Request,
    chat_session: ChatSession = Depends(get_chat_session)
):
    try:
        data = await request.json()
        user_query = data.get("query", "")

        if not isinstance(user_query, str) or not user_query.strip():
            return JSONResponse(content={"error": "Query is required."}, status_code=422)

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)

        try:
            history = await chat_session.get_history()
        except Exception as e:
            logger.error(f"Chat session history error: {e}")
            history = []

        history_text = "\n".join([
            f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            for msg in history
        ])

        intent_prompt = (
            "Classify the user's intent.\n\n"
            "If the question is related to library services (like books, borrowing, locations, library staff, etc.), reply with: library\n"
            "If the question is general, personal, or off-topic (like greetings, languages, preferences, abilities), reply with: general\n\n"
            f"User Question: {translated_query}\n"
            "Intent:"
        )
        try:
            intent_response = await generate_response(intent_prompt)
            intent = intent_response.strip().lower()
            logger.info(f"[Intent Detection] Query: '{translated_query}' → Intent: {intent}")
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")
            intent = "library"

        if intent == "general":
            prompt = library_fallback_prompt(history_text, translated_query)
            try:
                bot_response = await generate_response(prompt)
            except Exception as e:
                logger.error(f"LLM fallback error: {e}")
                bot_response = "Sorry, I couldn't respond at the moment."

            translated_response = await translate_to_user_language(bot_response, user_lang)
            response_content = format_response(translated_response, default_reminders)

            try:
                await chat_session.add_message("user", user_query)
                await chat_session.add_message("assistant", translated_response)
            except Exception as e:
                logger.error(f"Chat session log error: {e}")

            response_content["history"] = await chat_session.get_history()
            return JSONResponse(content=response_content, status_code=200)


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

        try:
            await chat_session.add_message("user", user_query)
        except Exception as e:
            logger.error(f"Chat session add_message (user) error: {e}")
        try:
            await chat_session.add_message("assistant", translated_response)
        except Exception as e:
            logger.error(f"Chat session add_message (assistant) error: {e}")

        history = await chat_session.get_history()
        response_content["history"] = history
        return JSONResponse(content=response_content, status_code=200)

    except Exception as e:
        logger.error(f"Library info endpoint failed: {e}")
        return JSONResponse(
            content={"error": "Internal server error."},
            status_code=500
        )
