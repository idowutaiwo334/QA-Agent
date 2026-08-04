# About This Agent

This is a sample document included so you can test the Q&A agent immediately
after setup, before adding your own content.

## How it works

The agent uses Retrieval-Augmented Generation (RAG). When you ask a question:

1. Your question is converted into a vector embedding.
2. The system searches a local vector database for the most relevant chunks
   of text from the documents in the `/data` folder.
3. Those chunks are sent to Claude along with your question.
4. Claude answers using only that retrieved context, and cites which source
   files it used.

## Adding your own documents

Delete or keep this file, then drop your own `.txt`, `.md`, or `.pdf` files
into the `/data` folder. After adding files, run:

    python ingest.py

from the `backend` folder, or call the `/api/ingest` endpoint if the server
is already running. This rebuilds the knowledge base from everything
currently in `/data`.

## Example questions to try right now

- "How does this agent work?"
- "What file types can I add to the data folder?"
- "How do I add new documents after the server is already running?"
