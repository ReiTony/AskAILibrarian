from utils.llm_client import generate_response
from utils.llm_intent_prompt import intent_classifier_prompt

async def classify_intent(user_query, history):
    prompt = intent_classifier_prompt(history, user_query)
    response = await generate_response(prompt)
    return response.strip().lower()
