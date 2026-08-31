# RoleReady AI

Mock interview practice: 10 scored questions, Streamlit UI, SQLite question bank, Pinecone retrieval, OpenAI interviewer and scorer.

API keys are never hardcoded. Copy `.env.example` to `.env` and fill in secrets locally. Do not commit `.env`.

## Setup

Use Python 3.11 or newer. Run every command from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`. Optional models, index name, and `SQLITE_PATH` are documented in `.env.example`.

If an existing Pinecone index is not the model's native size (1536 for `text-embedding-3-small`), set `OPENAI_EMBEDDING_DIMENSIONS` to the index dimension (for example `1024`).

## Seed SQLite and Pinecone

```powershell
python scripts/init_db.py
python scripts/ingest_pinecone.py
```

`init_db.py` does not call OpenAI or Pinecone. Ingest embeds the question bank and upserts vectors.

## Run the app

```powershell
streamlit run src/roleready/ui/app.py
```

OpenAI and Pinecone clients are created when you click **Start Interview**, not when the setup page first loads.

## Tests

```powershell
pytest
```

Unit tests use fakes. They do not call OpenAI or Pinecone and they do not write to `data/roleready.db`.
