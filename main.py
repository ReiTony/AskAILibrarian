import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from decouple import config
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import uvicorn

# ---- Global Logging Config (one place only) ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# App Imports
from routes.authentication_route import router as auth_router
from routes.chat_route import router as chat_router
from routes.library_info_route import router as library_info_route
from routes.librarian_route import router as search_books_router
from routes.query_router import router as query_router
from utils.chroma._chroma_init import initialize_chroma

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_chroma()
    yield

# FastAPI App Initialization
app = FastAPI(
    title="Koha Chatbot",
    description="Koha Library Chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics Monitoring
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(auth_router, prefix="/api/auth", tags=["User Authentication"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chatbot"])
app.include_router(library_info_route, prefix="/api/library", tags=["Library Information"])
app.include_router(search_books_router, prefix="/api/search", tags=["Book Search"])
app.include_router(query_router, prefix="/api/query", tags=["Query Processing"])

# Health Endpoints
@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


