# AskAILibriarian — Turnover Documentation

## Overview

AskAILibriarian is a Python-based assistant for library/book discovery and conversational interfaces. It combines a small web API with vector search (Chroma), a local SQLite-backed dataset of book/web embeddings, and optional Rasa-based conversational automation. This document is a guide describing the project layout, setup, run steps, major components, and troubleshooting notes.

---

## Table of Contents

- Project summary
- Quick start (run locally)
- Architecture & folder map
- Key components (routes, utils, db, chroma)
- Data & vector indices (Chroma details)
- Database populate scripts
- Routes / API endpoints
- Rasa (optional) integration & setup
- Testing and development workflow
- Troubleshooting & logs
- Security, deployment notes, and next steps

---

## Project summary

- Language: Python (3.10 recommended — project virtual envs exist for reference)
- Entrypoint: `main.py` (starts Flask/FastAPI-style server or custom runner — check `main.py` for exact server startup)
- Vector search: Chroma indices stored under `chroma/` (separate DB files for `books` and `web1`)
- Databases: SQLite files under `chroma/` subfolders and other DB utils in `db/`
- Optional conversational layer: Rasa (code under `rasa/`), configured to be optional so core API works without Rasa running

---

## Quick start — run locally (PowerShell)

These commands assume you're on Windows and want to use the provided venv. If you prefer to create a fresh venv, substitute accordingly.

Activate the app virtual environment (existing one in repo):

```powershell
# Use the included env (if that matches your machine) — PowerShell
.\venv_librarian\Scripts\Activate.ps1

# Or create one and activate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install requirements (use the provided file):

```powershell
pip install -r librarian_requirements.txt
```

Run the app:

```powershell
python main.py
```

If the app exposes an HTTP port, the console will show which port (commonly 8000/5000/3000 depending on framework). Open the base address in your browser or use curl/Postman to hit endpoints under `routes/`.

---

## Architecture & folder map

High-level layout (important files and purpose):

- `main.py` — application entrypoint. Start here to see the actual server used (Flask/FastAPI/other).
- `README.md` — project README (short);

Primary folders:
- `routes/` — HTTP routes; main API layer. Key files: `librarian_route.py`, `query_router.py`, `rasa_route.py`, `library_info_route.py`.
- `utils/` — helper modules: `llm_client.py`, `chroma_client.py`, `koha_client.py`, `text_utils.py`, `prompt_templates.py`, `sessions.py`, etc.
- `chroma/` — Chroma DB data and index folders
  - `chroma/books/` — `chroma.sqlite3` and subfolder(s) for embedding indexes
  - `chroma/web1/` — another chroma instance for web content
- `db/` — database connection helpers (e.g., `connection.py`)
- `db_populate/` — scripts used to populate chroma / sqlite indices:
  - `populate_database_books.py`
  - `populate_database_web.py`
- `rasa/` — optional Rasa assistant configuration and action server; includes `config.yml`, `domain.yml`, `endpoints.yml`, `actions/`.
- `models/` — model artifacts used by the app (small models, or containerized models). Validate content for exact usage.
- `deprecated/` — older code kept for reference. Not used in production.

Virtual environments (for convenience): `venv_librarian/` and `venv_rasa/` — provided virtual envs for librarian web app and rasa development respectively.

---

## Key components and responsibilities

- routes/* — translate HTTP requests to internal service calls and compose responses. They call into `utils/*` for heavy lifting.
- utils/llm_client.py — wrapper to call LLM providers (OpenAI or other). Responsible for prompt construction and sending requests.
- utils/chroma_client.py — wrapper for Chroma vector store calls (search, insert, delete). Use this to interact with indices under `chroma/`.
- db/connection.py — DB connection helper(s) to interact with sqlite (and any other DBs).
- db_populate/* — scripts to create or refresh embeddings and push them into Chroma.sqlite stores. Run these when you add new documents or need to rebuild indices.

Contract (how components talk):
- Input: HTTP JSON payloads (chat, query) -> routes
- Processing: routes call utils (embedding/LLM/chroma/db) and combine outputs
- Output: JSON responses (chat messages, recommendations, search results)

Edge cases to watch for:
- Missing environment variables for LLM keys or Chroma paths
- Token/quota limits from the LLM provider
- Large payloads causing long embedding times
- Concurrency when rebuilding indices while serving queries

---

## Data & vector indices (Chroma)

There are two Chromadb-backed folders in `chroma/`:
- `chroma/books/` — used for the book dataset. Contains `chroma.sqlite3` database and an index folder with id-based subfolders.
- `chroma/web1/` — web dataset; same structure.

To rebuild or populate indices, use `db_populate/populate_database_books.py` and `db_populate/populate_database_web.py`. Those scripts will:
- read source documents (format depends on script)
- compute embeddings via the project's embedding function (see `utils/chroma/_get_embedding_function.py` and `utils/chroma/_chroma_init.py`)
- insert or update vectors into the chroma SQLite store

Basic steps to rebuild (high-level):
1. Activate venv
2. Ensure env vars (LLM keys) are set
3. Run populate script:

```powershell
python db_populate\populate_database_books.py
```

Note: These scripts may assume a certain working directory. Run them from repo root to avoid path issues.

---

## Database notes

- Primary lightweight stores: SQLite files under `chroma/*` for vector store and `db/` for other app data.
- Backups: Make copies of `chroma/*/chroma.sqlite3` before destructive operations.

---

## Routes / Endpoints (summary)

The primary user-facing endpoint implemented in `routes/librarian_route.py` is:

- POST /search_books — main search / lookup / recommendation endpoint. It expects a JSON body with at least a `query` field. The route accepts an optional query parameter `intent` (e.g. `?intent=book_search`) which controls the flow. The endpoint also uses a `session-id` HTTP header (FastAPI dependency `get_session_id`) to bind short-term session memory.

Other routes live in `routes/` and should be inspected for full details:

- `routes/query_router.py` — additional query processing flows
- `routes/library_info_route.py` — library metadata endpoints
- `routes/rasa_route.py` — optional bridge to Rasa (if enabled)

Notes on dependency behavior and headers:
- `session-id` header: optional. If provided, it ties requests to an in-memory ChatSession (see `utils/sessions.py`). If omitted, the server will generate a session id and return it to the caller in responses or logs.
- Request body: parsed as JSON by `get_session_and_user_data`. Typical fields used by `/search_books` are `query` (string) and `cardNumber` (string — user library card / identifier). Additional fields may be accepted by other routes.

Detailed examples for `/search_books` (exact payloads and sample responses)

1) Book search (intent=book_search)

Request

POST /search_books?intent=book_search
Headers:
- Content-Type: application/json
- session-id: d290f1ee-6c54-4b01-90e6-d701748f0851  # optional

Body:

```json
{
  "cardNumber": "12345",
  "query": "science fiction time travel"
}
```

Successful response (200)

```json
{
  "response": [
    {
      "type": "booksearch",
      "answer": "Here are some books about science fiction and time travel.",
      "books": [
        {
          "title": "The Time Machine",
          "author": "H. G. Wells",
          "isbn": "9780451528551",
          "publisher": "Penguin",
          "year": "1895",
          "biblio_id": "101",
          "quantity_available": 3
        },
        {
          "title": "Kindred",
          "author": "Octavia E. Butler",
          "isbn": "9780446675505",
          "publisher": "Spectra",
          "year": "1979",
          "biblio_id": "102",
          "quantity_available": 1
        }
      ]
    }
  ]
}
```

Notes: the `books` array items follow the structure assembled inside `librarian_route.py` (`title`, `author`, `isbn`, `publisher`, `year`, `biblio_id`) and will include `quantity_available` after the route fetches item counts.

2) Book recommendation (intent=book_recommend)

Request

POST /search_books?intent=book_recommend
Headers:
- Content-Type: application/json

Body:

```json
{
  "cardNumber": "12345",
  "query": "books similar to Dune"
}
```

Successful response (200)

```json
{
  "response": [
    {
      "type": "recommendation",
      "books": [
        {
          "title": "Hyperion",
          "author": "Dan Simmons",
          "isbn": "9780553283686",
          "publisher": "Bantam",
          "year": "1989",
          "biblio_id": "210",
          "quantity_available": 2
        }
      ]
    }
  ]
}
```

Note: The route uses an LLM prompt (`recommend_books_prompt`) to craft the assistant reply and selects up to 10 unique books to return.

3) Specific book lookup by identifier (intent=book_lookup_isbn)

This flow attempts to extract ISBN / ISSN / call numbers from the `query` and will return either a `specific_book_search` response with a list of matching records, or a `specific_book_search` response with an explanatory `answer` when nothing is found.

Request

POST /search_books?intent=book_lookup_isbn
Headers:
- Content-Type: application/json

Body:

```json
{
  "cardNumber": "12345",
  "query": "ISBN 9780143127550"
}
```

Successful response (found)

```json
{
  "response": [
    {
      "type": "specific_book_search",
      "answer": "I found a copy of 'Sapiens' (ISBN 9780143127550) in the catalog.",
      "books": [
        {
          "title": "Sapiens: A Brief History of Humankind",
          "author": "Yuval Noah Harari",
          "isbn": "9780143127550",
          "publisher": "Harper",
          "year": "2015",
          "biblio_id": "305",
          "quantity_available": 4
        }
      ]
    }
  ]
}
```

Not found response (200 with explanatory answer)

```json
{
  "response": [
    {
      "type": "specific_book_search",
      "answer": "I couldn't find a record for ISBN/ISSN/Call Number.",
      "books": []
    }
  ]
}
```

4) Error cases

- Missing `query` in body -> 400

Request body:

```json
{
  "cardNumber": "12345"
}
```

Response (400):

```json
{
  "error": "Query is required."
}
```

- Unrecognized `intent` -> 400 with error message

5) How intent is passed

The `intent` parameter in the route function is not part of the JSON body in `librarian_route.py` and is expected to be passed as a query parameter, e.g. `/search_books?intent=book_search`. The main application or a higher-level router may set this automatically when bridging from Rasa or another dialog manager.

6) Session lifecycle and cardNumber binding

- If you provide `cardNumber` in the request body and the `session-id` header is present (or subsequently returned), the `get_session_and_user_data` dependency will attach the `cardNumber` to the in-memory `ChatSession` so subsequent calls in the same session will reuse the card number. See `utils/sessions.py` for details.

Add automated tests (or inspect `routes/*` for example payloads) to capture exact schema.

---

## Rasa (optional) integration

Rasa is included as an optional conversational layer. The project includes a `rasa/` folder with the following:
- `config.yml`, `domain.yml`, `credentials.yml`, `endpoints.yml`
- `rasa/actions/` -> custom action server code (`actions.py`) which may call the app's internal endpoints or utils

When to enable Rasa:
- You want a multi-turn, rule-based or ML-based dialogue manager that triggers the library assistant's capabilities
- You want to separate NLU/dialogue management (Rasa) from the data/embedding/LLM components

How to enable Rasa (high-level):
1. Activate the Rasa venv (or install Rasa into the app venv):

```powershell
# If using the provided venv_rasa:
.\venv_rasa\Scripts\Activate.ps1
pip install -r rasa\rasa_requirements.txt
```

2. Start the action server (if required by `endpoints.yml`):

```powershell
# From the repo root
python -m rasa.actions
```

3. Launch Rasa (core server):

```powershell
rasa run --endpoints rasa\endpoints.yml
# or during development you can run the interactive server
rasa shell
```

4. Configure `rasa_route.py` and `routes/rasa_*` to point to the Rasa endpoints (check `rasa/endpoints.yml` and `routes/rasa_route.py` for exact host/port).

Notes:
- Rasa may require additional model training: `rasa train` (this uses the `data/` files and `domain.yml`).
- Rasa's `actions.py` should be reviewed; it may call the app's internal endpoints. Ensure those calls use correct host/port and auth if required.

---

## Testing & developer workflow

- Unit tests: check for tests under `tests/` or `rasa/tests/`. Add targeted tests for route handlers (Flask/Starlette test client) and utils.
- Manual testing: use Postman, curl, or a small Python script to post requests to endpoints in `routes/`.
- Test embedding refresh: run `db_populate/*` scripts and then run a few sample queries to verify results.

Sample local dev workflow:
1. Activate venv
2. Run backend server (`python main.py`)
3. Optionally run Rasa (if you're testing dialogue flows)
4. Open Postman or use sample curl requests to exercise endpoints

---

## Troubleshooting & common issues

- Missing API keys / env vars: Check any code reading keys from environment. Common env vars include LLM provider keys and embedding model identifiers.
- Chroma locked or corrupted: If `chroma.sqlite3` is locked, stop the app and any scripts using it; make a backup before running repair/rebuild.
- Long indexing times: population scripts compute embeddings; throttling and batching help. Consider running populate scripts on a separate worker machine or as a background job.
- Rasa connection issues: Verify `endpoints.yml` and that `rasa.actions` server is running. Check ports and firewall.
- Unexpected LLM responses: Confirm prompt templates under `utils/prompt_templates.py` and LLM configuration in `utils/llm_client.py`.

Logging tips:
- Enable debug logs in `main.py` or route boot logic
- Inspect stdout/stderr for stack traces
- Use the sqlite browser to inspect chroma DB content if search returns empty

---

## Security and privacy notes

- Do not commit API keys or credentials. Use environment variables or a secrets manager.
- Chroma data may contain PII if web scrapes or book metadata includes it. Follow your org's data handling rules.

---

## Deployment notes

- For production: containerize the app (Docker) and mount volumes for Chroma sqlite files so indices persist.
- Scale LLM calls and Chroma reads with a horizontally scalable architecture: move Chroma to a service-backed store if needed, or shard indices by dataset.
- Rasa should run in its own container/host, with secure communication between Rasa and this app.

---

## Where to look in the code now (quick pointers)

- `main.py` — server startup
- `routes/librarian_route.py` — primary user-facing endpoints
- `utils/llm_client.py` — how LLM calls are made
- `utils/chroma_client.py` & `utils/chroma/_get_embedding_function.py` — embedding and vector storage
- `db_populate/` — scripts used to create indices
- `rasa/` — optional conversational assistant files

---

