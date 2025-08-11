import logging
from fastapi.responses import JSONResponse
from utils.llm_client import generate_response
from utils.chat_retention import get_retained_history
from utils.prompt_templates import library_fallback_prompt

logger = logging.getLogger("general_info_handler")

async def handle_general_info(session_data, db, **kwargs):
    chat_session, cardnumber, data = session_data
    user_query = data.get("query", "").strip()
    cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None)
    # Get recent chat history for context
    full_history = await get_retained_history(db, cardnumber) + await chat_session.get_history()
    history_text = "\n".join(
        f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
        for msg in full_history[-4:]
    )
    prompt = library_fallback_prompt(history_text, user_query)
    logger.info(f"Generating response for general info with prompt: {prompt}")
    reply = await generate_response(prompt)
    return JSONResponse(content={"answer": reply}, status_code=200)