import httpx
from decouple import config
import logging

OPENROUTER_API_KEY = config("OPENROUTER_API_KEY2")  
SITE_URL = config("SITE_URL", default="http://localhost")
SITE_TITLE = config("SITE_TITLE", default="Librarian Chatbot")

MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"
logger = logging.getLogger("llm_client")

async def generate_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_TITLE
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6,
        "top_p": 0.9
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenRouter API returned error: {response.status_code} - {response.text}")
            return "[ERROR]: The AI service is currently unavailable. Please try again later."
    except Exception as e:
        logger.error(f"OpenRouter API request failed: {str(e)}")
        return "[ERROR]: The AI service is currently unavailable. Please try again later."
