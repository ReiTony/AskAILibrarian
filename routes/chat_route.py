import logging
from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from bson import ObjectId

from utils.sessions import ChatSession, get_chat_session
from schemas.chat_schemas import Email, SessionId, Sender, MessageText, NewName, MessageIndex, NewText, DeleteSubsequent
from db.connection import get_db

logger = logging.getLogger("chat_route")
router = APIRouter()

def get_chat_collection(db: AsyncIOMotorDatabase):
    return db["chathistories"]

def clean_object_ids(obj):
    if isinstance(obj, list):
        return [clean_object_ids(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_object_ids(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    return obj

@router.post("/save-chat")
async def save_chat(
    email: Email,
    sessionId: SessionId,
    sender: Sender,
    message: MessageText,
    db = Depends(get_db),
    chat_session: ChatSession = Depends(get_chat_session),
):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"email": email})
        msg_obj = {"text": message, "sender": sender, "timestamp": datetime.utcnow()}

        await chat_session.add_message(sender, message)
        logger.info(f"[Session {sessionId}] [User: {email}] {sender.capitalize()} said: {message}")

        if not chat:
            new_chat = {
                "email": email,
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

        await chats.replace_one({"email": email}, chat)
        return {"message": "Chat saved successfully", "savedMessages": session["messages"] if session else [msg_obj]}
    except Exception as e:
        logger.error(f"Save chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.get("/get-chat-history")
async def get_chat_history(email: str, db = Depends(get_db)):
    try:
        chat = await get_chat_collection(db).find_one({"email": email})
        if not chat or "sessions" not in chat:
            return []
        cleaned_sessions = clean_object_ids(chat["sessions"])
        return cleaned_sessions
    except Exception as e:
        logger.error(f"Get chat history error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.delete("/delete-session/{email}/{sessionId}")
async def delete_session(email: str, sessionId: str, db = Depends(get_db)):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"email": email})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

        chat["sessions"] = [s for s in chat["sessions"] if s["sessionId"] != sessionId]
        await chats.replace_one({"email": email}, chat)
        return {"message": "Chat session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.put("/update-chat-name/{email}/{sessionId}")
async def update_chat_name(email: str, sessionId: str, newName: NewName, db = Depends(get_db)):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"email": email})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

        session = next((s for s in chat["sessions"] if s["sessionId"] == sessionId), None)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session["name"] = newName
        await chats.replace_one({"email": email}, chat)
        return {"message": "Chat name updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update chat name error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.put("/update-message/{email}/{sessionId}")
async def update_message(
    email: str,
    sessionId: str,
    messageIndex: MessageIndex,
    newText: NewText,
    deleteSubsequent: DeleteSubsequent,
    db = Depends(get_db)
):
    chats = get_chat_collection(db)
    try:
        chat = await chats.find_one({"email": email})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat history not found")

        session = next((s for s in chat["sessions"] if s["sessionId"] == sessionId), None)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if 0 <= messageIndex < len(session["messages"]):
            session["messages"][messageIndex]["text"] = newText
            if deleteSubsequent and messageIndex < len(session["messages"]) - 1:
                session["messages"] = session["messages"][:messageIndex + 1]

        await chats.replace_one({"email": email}, chat)
        return {"message": "Message updated successfully", "updatedMessages": session["messages"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update message error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
