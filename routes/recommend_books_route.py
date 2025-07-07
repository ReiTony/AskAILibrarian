from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import logging

from utils.prompt_templates import recommend_books_prompt 
from utils.sessions import ChatSession, get_chat_session
from utils.chroma_client import books_db
from utils.text_utils import replace_null
from utils.llm_client import generate_response
from utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language,
    extract_protected,
    restore_protected
)

router = APIRouter()
logger = logging.getLogger("recommend_books")

def clean_isbn(isbn):
    invalids = {"Unknown", "Not Available", "NO ISBN", "ISBN unavailable", ""}
    return isbn if isbn not in invalids else "ISBN unavailable"

book_recommend_suggestions = [
    "Search for more books by this author or subject.",
    "Recommend me another book.",
    "I need help with library borrowing or services?"
]

@router.post("/recommend")
async def recommend_books(
    request: Request,
    chat_session: ChatSession = Depends(get_chat_session)
):
    try:
        data = await request.json()
        user_query = data.get("query", "")

        user_lang = detect_language(user_query)

        translated_query_task = asyncio.create_task(translate_to_english(user_query, user_lang))
        history_task = chat_session.get_history()
        translated_query, history = await asyncio.gather(translated_query_task, history_task)

        try:
            results = books_db.similarity_search_with_score(translated_query, k=8)
        except Exception as e:
            logger.error(f"Error running similarity search: {e}")
            results = []

        book_list = []
        seen = set()
        for doc, _ in results:
            title = replace_null(doc.metadata.get("Title", "")).title()
            author = replace_null(doc.metadata.get("Author", ""))
            isbn = clean_isbn(replace_null(doc.metadata.get("ISBN", "")))
            key = (title.lower(), author.lower())
            if title and key not in seen:
                book_list.append({
                    "title": f"[[{title}]]",
                    "author": f"[[{author}]]",
                    "isbn": isbn
                })
                seen.add(key)
            if len(book_list) >= 8:
                break

        if book_list:
            prompt = recommend_books_prompt(translated_query, book_list)
            logger.info(f"Prompt to LLM (Library Books):\n{prompt}")
            try:
                bot_response = await generate_response(prompt)
                logger.debug(f"[LLM Raw Answer]: {bot_response}")
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                bot_response = "Sorry, I could not generate a response at this time."
            answer = bot_response
        else:
            answer = "Sorry, there are no suitable books in our library catalog for your query."

        protected_text, protected_map = extract_protected(answer.strip())
        logger.debug(f"[Protected Text]: {protected_text}")
        logger.debug(f"[Protected Map]: {protected_map}")
        translated_answer = await translate_to_user_language(protected_text, user_lang)
        translated_answer = restore_protected(translated_answer, protected_map)

        suggestions = book_recommend_suggestions

        try:
            await chat_session.add_message("user", user_query)
        except Exception as e:
            logger.error(f"Error adding user message to chat session: {e}")
        try:
            await chat_session.add_message("assistant", translated_answer)
        except Exception as e:
            logger.error(f"Error adding assistant message to chat session: {e}")

        return JSONResponse(
            content={
                "answer": translated_answer,
                "suggestion1": suggestions[0],
                "suggestion2": suggestions[1],
                "suggestion3": suggestions[2],
                "history": history,
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error occurred while processing /recommend: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
