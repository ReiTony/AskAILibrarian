import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger("chat_retention")

RETENTION_LIMIT = 15 
COLLECTION_NAME = "chat_retention_history"

async def save_conversation_turn(
    db: AsyncIOMotorDatabase,
    cardNumber: str,
    user_query: str,
    ai_response: str
):
    """
    This function is designed to be highly efficient by using MongoDB's atomic operators.
    It performs a single `find_one_and_update` operation with `upsert=True`.

    Args:
        db: The async motor database instance.
        cardNumber: The unique identifier for the user.
        user_query: The text of the user's message.
        ai_response: The text of the AI's generated response.
    """
    if not cardNumber or not user_query or not ai_response:
        logger.warning("[Chat Retention] Missing cardNumber, query, or response. Skipping save.")
        return

    collection = db[COLLECTION_NAME]
    
    # Prepare the two message documents for this turn
    messages_to_add = [
        {"role": "user", "content": user_query, "timestamp": datetime.utcnow()},
        {"role": "assistant", "content": ai_response, "timestamp": datetime.utcnow()}
    ]

    try:
        await collection.find_one_and_update(
            {"cardNumber": cardNumber},
            {
                "$push": {
                    "history": {
                        "$each": messages_to_add,
                        "$slice": -RETENTION_LIMIT  # Keep the last 30 messages
                    }
                },
                "$set": {"last_updated": datetime.utcnow()}
            },
            upsert=True  # Create the document if it doesn't exist
        )
        logger.info(f"[Chat Retention] Saved conversation turn for cardNumber: {cardNumber}")
    except Exception as e:
        logger.error(f"[Chat Retention] Failed to save history for {cardNumber}: {e}", exc_info=True)


async def get_retained_history(db: AsyncIOMotorDatabase, cardNumber: str) -> list[dict]:
    """
    Retrieves the retained conversation history for a given user.

    Args:
        db: The async motor database instance.
        cardNumber: The unique identifier for the user.

    Returns:
        A list of message dictionaries (e.g., [{'role': 'user', 'content': '...'}]) or an empty list.
    """
    if not cardNumber:
        return []

    collection = db[COLLECTION_NAME]
    try:
        document = await collection.find_one(
            {"cardNumber": cardNumber},
            {"history": 1, "_id": 0} # Projection to only return the history field
        )
        if document and "history" in document:
            logger.info(f"[Chat Retention] Fetched {len(document['history'])} messages for {cardNumber}")
            return document["history"]
        else:
            logger.info(f"[Chat Retention] No retained history found for {cardNumber}")
            return []
    except Exception as e:
        logger.error(f"[Chat Retention] Failed to fetch history for {cardNumber}: {e}", exc_info=True)
        return []