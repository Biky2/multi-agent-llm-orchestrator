# AgentFlow — Multi-Agent LLM Workflow Orchestrator

A production-ready pipeline of four specialized LLM agents (Researcher → Planner → Executor → Validator) that collaborate autonomously on complex tasks. Watch agents work in real time via SSE streaming. Sessions persist in Redis; completed runs are stored in PostgreSQL.

![AgentFlow Screenshot](docs/screenshot.png)

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (with `gen_random_uuid()` support)
- Redis 6+
- Ollama (optional, for local LLM) **or** HuggingFace API key (fallback)

## Installation

```bash
cd orchestrator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env .env.local          # optional: edit credentials
```

Create the PostgreSQL database:

```bash
createdb orchestrator
```

Edit `.env` with your connection strings and API keys.

## Running with Ollama (primary)

1. Install and start Ollama: https://ollama.com
2. Pull the model:

```bash
ollama pull llama3.2
```

3. Start Redis and PostgreSQL locally.
4. Run the server:

```bash
cd orchestrator
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

5. Open http://localhost:8000 in your browser.

If Ollama is unreachable, the system automatically falls back to HuggingFace Inference API when `HUGGINGFACE_API_KEY` is set.

## Running with HuggingFace fallback only

1. Stop Ollama or set `OLLAMA_BASE_URL` to an invalid host.
2. Set a valid `HUGGINGFACE_API_KEY` in `.env`.
3. Start the server as above.

The agents use the unified `get_llm_response()` abstraction and do not know which backend is active.

## API Examples

### Submit a task

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a brief market analysis of electric vehicle adoption in Europe"}'
```

Response:

```json
{
  "session_id": "a1b2c3d4-...",
  "stream_url": "/stream/a1b2c3d4-..."
}
```

### Stream agent events (SSE)

```bash
curl -N http://localhost:8000/stream/{session_id}
```

### Get session history

```bash
curl http://localhost:8000/history/{session_id}
```

### Health check

```bash
curl http://localhost:8000/health
```

## Architecture

```
User → FastAPI → LangGraph Pipeline
                    ├── Researcher
                    ├── Planner
                    ├── Executor
                    └── Validator (retry loop if confidence < 0.6)
         ↓ SSE Queue (real-time events)
         ↓ Redis (session state, 1h TTL)
         ↓ PostgreSQL (task_history)
```

## Project Structure

```
orchestrator/
  agents/          # Researcher, Planner, Executor, Validator
  graph/           # LangGraph workflow + state schema
  memory/          # Redis + PostgreSQL clients
  core/            # LLM abstraction (Ollama + HuggingFace)
  api/             # FastAPI routes, SSE, main app
  static/          # Single-page frontend
  config/          # pydantic-settings
```

## License

MIT
