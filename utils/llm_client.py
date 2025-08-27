import httpx
from decouple import config
import logging
from groq import AsyncGroq, GroqError

client = AsyncGroq(
    api_key=config("GROQ1"),
)

SITE_URL = config("SITE_URL", default="http://localhost")
SITE_TITLE = config("SITE_TITLE", default="Librarian Chatbot")

MODEL_NAME = "openai/gpt-oss-20b"
logger = logging.getLogger("llm_client")

async def generate_response(prompt: str) -> str:

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
            temperature=0.6,
            max_tokens=2024,
            top_p=0.9,
            reasoning_format="hidden",
            reasoning_effort="low",
        )

        return chat_completion.choices[0].message.content
    except GroqError as e:
        logger.error(f"Groq API error: {e.__class__.__name__} - {e}")
        return "[ERROR]: The AI service returned an error. Please check the logs."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return (
            "[ERROR]: The AI service is currently unavailable. Please try again later."
        )
