from fastapi import APIRouter, HTTPException, Depends
from schemas.form_schemas import CardNumber, Password
from motor.motor_asyncio import AsyncIOMotorDatabase
from jose import jwt
from datetime import datetime, timedelta
import bcrypt, logging

from decouple import config
from db.connection import get_db

router = APIRouter()
logger = logging.getLogger("user_routes")

# ENV
JWT_SECRET = config("JWT_SECRET")

# MongoDB accessor
def get_user_collection(db: AsyncIOMotorDatabase):
    return db["users"]


# ----------------------- USER LOGIN   -----------------------
@router.post("/login")
async def login(cardnumber: CardNumber, password: Password, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({"cardnumber": cardnumber})
        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = jwt.encode(
            {"id": str(user["_id"]), "exp": datetime.utcnow() + timedelta(hours=1)},
            JWT_SECRET,
            algorithm="HS256"
        )
        return {"token": token, "cardnumber": user["cardnumber"], "username": user["username"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


