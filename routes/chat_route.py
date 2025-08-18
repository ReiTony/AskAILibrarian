import logging
from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from bson import ObjectId

from utils.sessions import ChatSession, get_chat_session
from schemas.chat_schemas import (
    CardNumber, SessionId, Sender, MessageText,
    NewName, MessageIndex, NewText, DeleteSubsequent
)
from db.connection import get_db

logger = logging.getLogger("chat_route")
router = APIRouter()

def get_chat_collection(db: AsyncIOMotorDatabase):
    return db["chat_retention_history"]

def clean_object_ids(obj):
    if isinstance(obj, list):
        return [clean_object_ids(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_object_ids(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    return obj

# --------------------------------------------
# Save message to chat session
# --------------------------------------------
@router.post("/save-chat")
async def save_chat(
    cardnumber: CardNumber,
    sessionId: SessionId,
    sender: Sender,
    message: MessageText,
    db = Depends(get_db),
    chat_session: ChatSession = Depends(get_chat_session),
):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"cardnumber": cardnumber})
        msg_obj = {"text": message, "sender": sender, "timestamp": datetime.utcnow()}

        await chat_session.add_message(sender, message)
        logger.info(f"[Session {sessionId}] [User: {cardnumber}] {sender.capitalize()} said: {message}")

        if not chat:
            new_chat = {
                "cardnumber": cardnumber,
                "sessions": [{
                    "sessionId": sessionId,
                    "name": None,
                    "messages": [msg_obj],
                    "startTime": datetime.utcnow()
                }]
            }
            await chats.insert_one(new_chat)
            recent_history = await chat_session.get_history()
            logger.info(f"[Session {sessionId}] Memory bubble after save: {recent_history}")
            return {"message": "Chat saved successfully", "savedMessages": [msg_obj]}

        session = next((s for s in chat["sessions"] if s["sessionId"] == sessionId), None)
        if session:
            session["messages"].append(msg_obj)
        else:
            chat["sessions"].append({
                "sessionId": sessionId,
                "name": None,
                "messages": [msg_obj],
                "startTime": datetime.utcnow()
            })

        await chats.replace_one({"cardnumber": cardnumber}, chat)
        return {"message": "Chat saved successfully", "savedMessages": session["messages"] if session else [msg_obj]}
    except Exception as e:
        logger.error(f"Save chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --------------------------------------------
# Get chat history by cardnumber
# --------------------------------------------
@router.get("/get-chat-history")
async def get_chat_history(cardnumber: str, db = Depends(get_db)):
    try:
        logger.info(f"Fetching chat history for card number: {cardnumber}")
        chat = await get_chat_collection(db).find_one({"cardnumber": cardnumber})
        
        if not chat:
            logger.info(f"No chat found for card number: {cardnumber}")
            return None
            
        # Remove sessions before returning
        chat.pop('sessions', None)
        cleaned_chat = clean_object_ids(chat)
        return cleaned_chat
        
    except Exception as e:
        logger.error(f"Get chat history error for {cardnumber}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

# --------------------------------------------
# Delete chat session by session ID
# --------------------------------------------
@router.delete("/delete-session/{cardnumber}/{sessionId}")
async def delete_session(cardnumber: str, sessionId: str, db = Depends(get_db)):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"cardnumber": cardnumber})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

        chat["sessions"] = [s for s in chat["sessions"] if s["sessionId"] != sessionId]
        await chats.replace_one({"cardnumber": cardnumber}, chat)
        return {"message": "Chat session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --------------------------------------------
# Rename a chat session
# --------------------------------------------
@router.put("/update-chat-name/{cardnumber}/{sessionId}")
async def update_chat_name(cardnumber: str, sessionId: str, newName: NewName, db = Depends(get_db)):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"cardnumber": cardnumber})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

        session = next((s for s in chat["sessions"] if s["sessionId"] == sessionId), None)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session["name"] = newName
        await chats.replace_one({"cardnumber": cardnumber}, chat)
        return {"message": "Chat name updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update chat name error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --------------------------------------------
# Update a specific message in a chat session
# --------------------------------------------
@router.put("/update-message/{cardnumber}/{sessionId}")
async def update_message(
    cardnumber: str,
    sessionId: str,
    messageIndex: MessageIndex,
    newText: NewText,
    deleteSubsequent: DeleteSubsequent,
    db = Depends(get_db)
):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"cardnumber": cardnumber})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

        session = next((s for s in chat["sessions"] if s["sessionId"] == sessionId), None)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if 0 <= messageIndex < len(session["messages"]):
            session["messages"][messageIndex]["text"] = newText
            if deleteSubsequent and messageIndex < len(session["messages"]) - 1:
                session["messages"] = session["messages"][:messageIndex + 1]

        await chats.replace_one({"cardnumber": cardnumber}, chat)
        return {"message": "Message updated successfully", "updatedMessages": session["messages"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update message error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
