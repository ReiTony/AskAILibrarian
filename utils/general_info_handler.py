from fastapi.responses import JSONResponse
from utils.llm_client import generate_response
from utils.prompt_templates import library_fallback_prompt

async def handle_general_info(session_data, db, **kwargs):
    chat_session, cardnumber, data = session_data
    user_query = data.get("query", "").strip()
    # Get recent chat history for context
    history = await chat_session.get_history()
    history_text = "\n".join(
        f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
        for msg in history[-4:]
    )
    prompt = library_fallback_prompt(history_text, user_query)
    reply = await generate_response(prompt)
    return JSONResponse(content={"answer": reply}, status_code=200)
