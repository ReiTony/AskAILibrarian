import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict

logger = logging.getLogger("chat_retention")

RETENTION_LIMIT = 15
COLLECTION_NAME = "chat_retention_history"

async def save_conversation_turn(
    db: AsyncIOMotorDatabase,
    cardnumber: str,
    user_query: str,
    ai_response: str
) -> None:
    """
    Save a pair of user and assistant messages to chat history.
    Keeps only the last RETENTION_LIMIT messages.

    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance.
        cardnumber (str): User identifier.
        user_query (str): Text input from the user.
        ai_response (str): AI-generated response.
    """
    if not all([cardnumber, user_query, ai_response]):
        logger.warning("[Chat Retention] Missing data — skipping save.")
        return

    collection = db[COLLECTION_NAME]
    timestamp = datetime.utcnow()

    messages = [
        {"role": "user", "content": user_query, "timestamp": timestamp},
        {"role": "assistant", "content": ai_response, "timestamp": timestamp},
    ]

    try:
        await collection.find_one_and_update(
            {"cardnumber": cardnumber},
            {
                "$push": {
                    "history": {
                        "$each": messages,
                        "$slice": -RETENTION_LIMIT
                    }
                },
                "$set": {"last_updated": timestamp}
            },
            upsert=True
        )
        logger.info(f"[Chat Retention] Saved turn for {cardnumber} ({len(messages)} messages).")
    except Exception as e:
        logger.error(f"[Chat Retention] Error saving for {cardnumber}: {e}", exc_info=True)


async def get_retained_history(
    db: AsyncIOMotorDatabase,
    cardnumber: str
) -> List[Dict[str, str]]:
    """
    Retrieve the last RETENTION_LIMIT messages for a user.

    Args:
        db (AsyncIOMotorDatabase): MongoDB database instance.
        cardNumber (str): User identifier.

    Returns:
        List[Dict]: Message history (role, content, timestamp).
    """
    if not cardnumber:
        logger.warning("[Chat Retention] Missing cardnumber — returning empty history.")
        return []

    collection = db[COLLECTION_NAME]
    try:
        document = await collection.find_one(
            {"cardnumber": cardnumber},
            {"history": 1, "_id": 0}
        )
        history = document.get("history", []) if document else []
        logger.info(f"[Chat Retention] Retrieved {len(history)} messages for {cardnumber}.")
        return history
    except Exception as e:
        logger.error(f"[Chat Retention] Error fetching for {cardnumber}: {e}", exc_info=True)
        return []
