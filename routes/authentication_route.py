from fastapi import APIRouter, HTTPException, Depends
from schemas.form_schemas import Email, Username, Password, Code, EmailOrUsername, ResetToken, NewPassword
from motor.motor_asyncio import AsyncIOMotorDatabase
from jose import jwt
import random, logging
from datetime import datetime, timedelta
import bcrypt

from utils.mailer import send_email
from decouple import config
from db.connection import get_db

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("user_routes")

# ENV
JWT_SECRET = config("JWT_SECRET")
GMAIL_USER = config("GMAIL_USER")
GMAIL_PASS = config("GMAIL_PASS")

# MongoDB accessor
def get_user_collection(db: AsyncIOMotorDatabase):
    return db["users"]

def generate_code():
    return str(random.randint(100000, 999999))

# Signup Or User Registration
@router.post("/signup")
async def signup(email: Email, username: Username, password: Password, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        if await users.find_one({"username": username}):
            raise HTTPException(status_code=400, detail="Username already exists")
        if await users.find_one({"email": email}):
            raise HTTPException(status_code=400, detail="Email already exists")

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        verification_code = generate_code()
        expires = datetime.utcnow() + timedelta(hours=1)

        user = {
            "email": email,
            "username": username,
            "password": hashed_pw.decode(),
            "verificationCode": verification_code,
            "verificationCodeExpires": expires,
            "isVerified": False,
        }
        await users.insert_one(user)
        try:
            send_email(email, "Email Verification", f"Your code is {verification_code}")
        except Exception as mail_err:
            logger.error(f"Email send failed for signup: {mail_err}")
            await users.delete_one({"email": email})
            raise HTTPException(status_code=500, detail="Failed to send verification email.")
        return {"message": "Verification code sent to email"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Email Verification
@router.post("/verify")
async def verify_email(email: Email, code: Code, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({
            "email": email,
            "verificationCode": code,
            "verificationCodeExpires": {"$gt": datetime.utcnow()}
        })
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired verification code")
        await users.update_one(
            {"_id": user["_id"]},
            {"$set": {"isVerified": True}, "$unset": {"verificationCode": "", "verificationCodeExpires": ""}}
        )
        return {"message": "Email verified successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify email error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# User Login
@router.post("/login")
async def login(emailOrUsername: EmailOrUsername, password: Password, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({
            "$or": [{"email": emailOrUsername}, {"username": emailOrUsername}]
        })
        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not user.get("isVerified", False):
            raise HTTPException(status_code=403, detail="Please verify your email before logging in")

        token = jwt.encode(
            {"id": str(user["_id"]), "exp": datetime.utcnow() + timedelta(hours=1)},
            JWT_SECRET,
            algorithm="HS256"
        )
        return {"token": token, "email": user["email"], "username": user["username"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Forgot Password (Sends reset token to email)
@router.post("/forgot-password")
async def forgot_password(email: Email, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({"email": email})
        if user:
            reset_token = generate_code()
            await users.update_one(
                {"_id": user["_id"]},
                {"$set": {
                    "resetPasswordToken": reset_token,
                    "resetPasswordExpires": datetime.utcnow() + timedelta(hours=1)
                }}
            )
            try:
                send_email(email, "Password Reset Request", f"Reset token: {reset_token}")
            except Exception as mail_err:
                logger.error(f"Email send failed for forgot-password: {mail_err}")
                pass
        return {"message": "If the email exists, a reset token was sent."}
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Reset Password (Accepts reset token and new password)
@router.post("/reset-password")
async def reset_password(token: ResetToken, newPassword: NewPassword, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({
            "resetPasswordToken": token,
            "resetPasswordExpires": {"$gt": datetime.utcnow()}
        })
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired token")

        hashed_pw = bcrypt.hashpw(newPassword.encode(), bcrypt.gensalt())
        await users.update_one(
            {"_id": user["_id"]},
            {"$set": {"password": hashed_pw.decode()}, "$unset": {"resetPasswordToken": "", "resetPasswordExpires": ""}}
        )
        return {"message": "Password has been reset successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Resend Verification Code
@router.post("/resend-verification")
async def resend_verification(email: Email, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({"email": email})
        if user and not user.get("isVerified", False):
            last_code_time = user.get("verificationCodeExpires")
            if last_code_time and (datetime.utcnow() - (last_code_time - timedelta(hours=1))).total_seconds() < 180:
                time_left = int(180 - (datetime.utcnow() - (last_code_time - timedelta(hours=1))).total_seconds())
                raise HTTPException(status_code=429, detail=f"Please wait {time_left} seconds before requesting a new code")
            verification_code = generate_code()
            await users.update_one(
                {"_id": user["_id"]},
                {"$set": {
                    "verificationCode": verification_code,
                    "verificationCodeExpires": datetime.utcnow() + timedelta(hours=1)
                }}
            )
            try:
                send_email(email, "Email Verification Code Resent", f"Your new verification code is: {verification_code}")
            except Exception as mail_err:
                logger.error(f"Email send failed for resend-verification: {mail_err}")
                pass
        return {"message": "If the email is registered and unverified, a new verification code was sent."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resend verification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Check Cooldown for Resending Verification Code
@router.get("/check-cooldown/{email}")
async def check_cooldown(email: str, db = Depends(get_db)):
    users = get_user_collection(db)
    try:
        user = await users.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        last_code_time = user.get("verificationCodeExpires")
        if last_code_time and (datetime.utcnow() - (last_code_time - timedelta(hours=1))).total_seconds() < 180:
            time_left = int(180 - (datetime.utcnow() - (last_code_time - timedelta(hours=1))).total_seconds())
            return {"timeLeft": time_left}
        return {"timeLeft": 0}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check cooldown error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Update User Profile
@router.put("/update-profile/{currentEmail}")
async def update_profile(
    currentEmail: str,
    email: Email,
    username: Username,
    password: Password,
    db = Depends(get_db)
):
    users = get_user_collection(db)
    try:
        user = await users.find_one({"email": currentEmail})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if email and email != currentEmail:
            if await users.find_one({"email": email}):
                raise HTTPException(status_code=409, detail="Email already exists")
            user["email"] = email

        if username:
            user["username"] = username

        if password:
            user["password"] = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        await users.replace_one({"_id": user["_id"]}, user)
        return {"message": "Profile updated successfully", "email": user["email"], "username": user["username"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
