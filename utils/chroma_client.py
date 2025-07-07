from langchain_chroma import Chroma
from utils.chroma._get_embedding_function import get_embedding_function
import logging

try:
    embedding_function = get_embedding_function()
    books_db = Chroma(
        persist_directory="chroma/books",
        embedding_function=embedding_function
    )
    web_db = Chroma(
        persist_directory="chroma/web1",
        embedding_function=embedding_function
    )
    logging.info("Chroma clients initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Chroma clients: {e}")
    raise
