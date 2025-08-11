import logging

logger = logging.getLogger("intent_classifier")

from utils.llm_client import generate_response
from utils.llm_intent_prompt import intent_classifier_prompt

async def classify_intent(user_query, history):
    prompt = intent_classifier_prompt(history, user_query)
    logger.info(f"Classifying intent for query: '{user_query}' with history: '{history}'")
    response = await generate_response(prompt)
    return response.strip().lower()