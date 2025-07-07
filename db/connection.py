from motor.motor_asyncio import AsyncIOMotorClient
from decouple import config

MONGO_URI = config("MONGO_URI")

client = AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()

def get_db():
    return db