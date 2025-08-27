# AskAILibrarian

AskAILibrarian is a FastAPI-powered chatbot that answers library questions, searches the Koha catalog, and recommends books. It relies on Groq LLMs, MongoDB, and Chroma for semantic search.

---

## Features

- **Groq LLMs** for natural-language understanding and responses
- **MongoDB** for user accounts and chat retention
- **Chroma (Vector DB)** for semantic search over internal documents
- **Koha REST API** for real-time catalog lookups
- Modular intent routing and extensible utility modules

---

## Project Layout

```
AskAILibrarian/
├── main.py                 # FastAPI app setup
├── db/                     # MongoDB helpers
├── routes/                 # API routes and intent dispatching
├── schemas/                # Pydantic models
└── utils/                  # LLM, Koha and Chroma helpers
```

> **Note:** `utils/chroma/_chroma_init.py` and `_get_embedding_function.py` are imported but absent; ensure these exist for Chroma initialization.

---

## Requirements

- Python **3.11+**
- A running **MongoDB** instance
- API credentials for:
  - Koha REST API (`KOHA_API`, `KOHA_USERNAME`, `KOHA_PASSWORD`)
  - Groq API (`GROQ1`)
  - Chroma embeddings
- spaCy model: `en_core_web_sm`

### Environment Variables

Create a `.env` file or otherwise supply the following variables:

| Variable | Description |
|----------|-------------|
| `MONGO_URI` | MongoDB connection string |
| `JWT_SECRET` | Secret for JWT tokens |
| `KOHA_API`, `KOHA_USERNAME`, `KOHA_PASSWORD` | Koha REST API credentials |
| `GROQ1` | Groq API key |
| `SITE_URL`, `SITE_TITLE` | (Optional) metadata for prompts |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

The application also exposes health checks at `GET /` and `GET /health`.

---

## API Overview

### Authentication – `/api/auth`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | POST   | Authenticate with `cardnumber` & `password`; returns JWT |

### Chat History – `/api/chat`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/save-chat`                         | POST   | Persist a chat message |
| `/get-chat-history`                  | GET    | List chat sessions (no messages) |
| `/delete-session/{cardnumber}/{id}`  | DELETE | Remove a chat session |
| `/update-chat-name/{cardnumber}/{id}`| PUT    | Rename a session |
| `/update-message/{cardnumber}/{id}`  | PUT    | Edit/truncate a message |

### Query Router – `/api/query`

1. Builds recent + retained history.
2. Runs intent classification (`utils.intent_classifier`).
3. Dispatches to the handler mapped in `INTENT_DISPATCH`.

### Librarian Search – `/api/search`

Handles intents:
- `book_search` – keyword search over Koha
- `book_recommend` – recommendation flow
- `book_lookup_isbn` – direct ISBN/ISSN lookup

### Library Info – `/api/library`

- Retrieves policy/location info via Chroma vector search.
- Generates reminders or suggestions based on keywords.
- Saves conversation via `utils.chat_retention`.

---

## Chat Retention & Sessions

- **In-memory:** `utils.sessions.ChatSession` keeps last 10 messages per session.
- **Persistent:** `utils.chat_retention` stores up to 15 recent turns per user in MongoDB (`chat_retention_history` collection).

---

## Extending the Project

1. **Add a New Intent**
   - Update classification rules in `utils/llm_intent_prompt.py`.
   - Implement the handler (new module or route).
   - Register it in `routes/query_router.py`’s `INTENT_DISPATCH`.

2. **Integrate Additional Data Sources**
   - Create a utility module for the new source.
   - Incorporate results into relevant prompts or handlers.

3. **Customize Authentication/Authorization**
   - Modify `routes/authentication_route.py` and token logic as needed.

---

## Contributing

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies.
3. Follow the existing code organization in `routes/` and `utils/`.
4. Submit pull requests with clear descriptions and adhere to coding style.

---

## License

Distributed under the [MIT License](LICENSE).

---
