# Machine-Vision

Semiconductor yield and defect detection project, built on MongoDB's
[smf-yield-defect-detection](https://github.com/mongodb-industry-solutions/smf-yield-defect-detection)
reference solution (MIT licensed, see `LICENSE`).

The upstream documentation is kept in [`README.upstream.md`](README.upstream.md). It covers
the architecture, Atlas vector and search indexes, and the setup steps.

## Quick start

```bash
cp .env.example .env        # fill in MONGODB_URI, AWS credentials, VOYAGE_API_KEY

# backend
cd backend && uv sync && uv run uvicorn main:app --reload --port 8000

# frontend (new terminal)
cd frontend && npm install && npm run dev
```

Or run both with `docker compose up --build`.
