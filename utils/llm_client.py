import httpx
from decouple import config
import logging

OPENROUTER_API_KEYS = [k.strip() for k in config("OPENROUTER_API_KEYS").split(",")]
SITE_URL = config("SITE_URL", default="http://localhost")
SITE_TITLE = config("SITE_TITLE", default="Librarian Chatbot")

MODEL_NAME = "mistralai/mistral-small-3.2-24b-instruct:free"
logger = logging.getLogger("llm_client")

def mask_key(key: str, start=4, end=3):
    """Show only the first `start` and last `end` characters of the API key."""
    return f"{key[:start]}...{key[-end:]}" if len(key) > (start + end) else key

async def generate_response(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.9
    }

    for key in OPENROUTER_API_KEYS:
        masked = mask_key(key)
        logger.info(f"Using OpenRouter API key: {masked}")

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_TITLE
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
            if response.status_code == 200:
                logger.info(f"OpenRouter request succeeded with key {masked}")
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code in (401, 403, 429):
                logger.warning(f"API key {masked} failed ({response.status_code}). Trying next key.")
                continue  # Try the next key
            else:
                logger.error(f"OpenRouter API returned error for key {masked}: {response.status_code} - {response.text}")
                break
        except Exception as e:
            logger.error(f"OpenRouter API request failed for key {masked}: {str(e)}")
            continue

    return "[ERROR]: The AI service is currently unavailable. Please try again later."
