# Q&A Agent

A retrieval-augmented Q&A agent: drop documents into `/data`, ask questions
in the web chat UI, and get answers grounded in those documents (with
sources cited). Runs as a single FastAPI service — no separate frontend
deployment needed.

- **Retrieval**: local vector search (Chroma), embeddings run locally via
  `sentence-transformers` — no extra API key needed for that part.
- **Generation**: Claude, via the Anthropic API.
- **Frontend**: a single static chat page, served by the same backend.

## Project structure

```
qa-agent/
├── backend/
│   ├── main.py          # FastAPI app (API + serves the frontend)
│   ├── rag.py            # chunking, embedding, retrieval, generation
│   ├── ingest.py          # CLI script: rebuild the vector DB from /data
│   ├── requirements.txt
│   └── .env.example
├── data/                  # put your .txt / .md / .pdf files here
├── frontend/
│   └── index.html         # chat UI
├── Procfile                # tells Railway/Render how to run it
└── runtime.txt
```

## 1. Run it locally

```bash
cd qa-agent/backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then edit .env and add your ANTHROPIC_API_KEY
python ingest.py                  # builds the vector DB from /data (sample.md is included)
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — you'll see the chat UI. Try asking:
*"How does this agent work?"*

Get an Anthropic API key at https://console.anthropic.com/settings/keys if
you don't have one.

## 2. Add your own documents

Put `.txt`, `.md`, or `.pdf` files into `/data` (delete `sample.md` if you
don't want it), then re-run:

```bash
python ingest.py
```

Or, if the server is already running, `POST` to `/api/ingest` to rebuild
without restarting.

## 3. Push to GitHub

```bash
cd qa-agent
git init
git add .
git commit -m "Initial commit: Q&A agent"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Create the empty repo on GitHub first, or use `gh repo create` if you have
the GitHub CLI installed.) `.env` is git-ignored, so your API key will never
be committed.

## 4. Deploy

### Option A — Railway (recommended, simplest)

1. Go to https://railway.app → **New Project** → **Deploy from GitHub repo**
   → select this repo.
2. Railway auto-detects the `Procfile` and `runtime.txt`.
3. In **Variables**, add `ANTHROPIC_API_KEY` = your key.
4. Deploy. Railway gives you a public URL — that's your live agent.

### Option B — Render

1. Go to https://render.com → **New** → **Web Service** → connect this repo.
2. Build command: `pip install -r backend/requirements.txt`
3. Start command: `cd backend && python ingest.py && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable `ANTHROPIC_API_KEY`.
5. Deploy.

Either way: whenever you push new documents to `/data` and commit/push,
redeploy to re-ingest them (the start command re-runs `ingest.py` on boot).

## Notes & next steps

- **Larger document sets**: the current setup chunks and re-embeds everything
  on every ingest, which is fine for up to a few hundred documents. For a
  large, frequently-changing corpus, consider incremental ingestion and a
  managed vector DB (e.g. Pinecone) instead of local Chroma.
- **Auth**: there's no login on the chat page — anyone with the URL can use
  it (and consume your Anthropic API credits). Add basic auth or an API key
  check in `main.py` before sharing the link widely.
- **Cost control**: each question makes one Claude API call. Watch usage at
  https://console.anthropic.com/settings/usage.
