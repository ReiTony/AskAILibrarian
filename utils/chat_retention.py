import logging
from datetime import datetime
from typing import List, Dict, Any

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger("chat_retention")

RECENT_MESSAGES_LIMIT = 15

async def save_message_and_get_context(
    db: AsyncIOMotorDatabase,
    cardnumber: str,
    session_id: str,
    sender: str,
    message_text: str,
) -> List[Dict[str, Any]]:

    chats_collection = db["chathistories"]
    msg_obj = {"text": message_text, "sender": sender, "timestamp": datetime.utcnow()}
    
    logger.info(
        f"[Chat Retention] Attempting to save message for cardnumber='{cardnumber}', "
        f"sessionId='{session_id}'"
    )

    try:
        # --- Step 1: Save the new message into the correct session ---

        # First, try to push the message to an existing session.
        # The positional '$' operator ensures we update the correct session in the array.
        update_result = await chats_collection.update_one(
            {"cardnumber": cardnumber, "sessions.sessionId": session_id},
            {"$push": {"sessions.$.messages": msg_obj}},
        )

        # If no document was matched, it means the session doesn't exist yet.
        # We need to add it to the user's document.
        if update_result.matched_count == 0:
            logger.warning(
                f"[Chat Retention] Session '{session_id}' not found for cardnumber '{cardnumber}'. "
                "Attempting to create a new session entry."
            )
            # This operation will add a new session object to the 'sessions' array.
            # 'upsert=True' will create the entire document if the cardnumber is new.
            await chats_collection.update_one(
                {"cardnumber": cardnumber},
                {
                    "$push": {
                        "sessions": {
                            "sessionId": session_id,
                            "name": None,  # A default name can be set later
                            "messages": [msg_obj],
                            "startTime": datetime.utcnow(),
                        }
                    }
                },
                upsert=True,
            )
        
        logger.info(f"[Chat Retention] Message successfully saved for sessionId: '{session_id}'")

        # --- Step 2: Retrieve the last 15 messages for context using an aggregation pipeline ---
        
        # This is the most efficient way to query a slice of a nested array in MongoDB.
        pipeline = [
            # Find the document for the specific user
            {"$match": {"cardnumber": cardnumber}},
            # Deconstruct the 'sessions' array to process each session individually
            {"$unwind": "$sessions"},
            # Filter down to the one session we care about
            {"$match": {"sessions.sessionId": session_id}},
            # Reshape the output to only give us the last N messages
            {
                "$project": {
                    "_id": 0,
                    "recent_messages": {
                        "$slice": ["$sessions.messages", -RECENT_MESSAGES_LIMIT]
                    },
                }
            },
        ]

        cursor = chats_collection.aggregate(pipeline)
        # We expect only one result, so we convert the cursor to a list of size 1
        result_list = await cursor.to_list(length=1)

        if not result_list:
            logger.warning(
                f"[Chat Retention] Could not retrieve context for "
                f"cardnumber='{cardnumber}', sessionId='{session_id}' after save."
            )
            return []

        recent_messages = result_list[0].get("recent_messages", [])
        logger.info(
            f"[Chat Retention] Retrieved {len(recent_messages)} recent messages for context."
        )
        return recent_messages

    except Exception as e:
        logger.error(
            f"Error in save_message_and_get_context for cardnumber='{cardnumber}', "
            f"sessionId='{session_id}': {e}",
            exc_info=True, # Set to True to log the full traceback for debugging
        )
        return []