import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
MISTRAL_BASE = "https://api.mistral.ai/v1"
HN_BASE = "https://hacker-news.firebaseio.com/v0"

HN_KEYWORDS = {
    "data", "ml", "ai", "llm", "mlops", "dbt", "spark", "kafka", "flink",
    "airflow", "pipeline", "warehouse", "lakehouse", "vector", "embedding",
    "model", "training", "inference", "gpu", "transformer", "rag",
    "openai", "anthropic", "mistral", "gemini", "huggingface",
    "databricks", "snowflake", "duckdb", "polars", "pandas", "python",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


class BriefRequest(BaseModel):
    title: str
    source: str
    tags: list[str]
    why: str


async def fetch_hn_news() -> list[dict]:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{HN_BASE}/topstories.json")
        resp.raise_for_status()
        story_ids = resp.json()[:100]

        import asyncio
        async def fetch_item(sid: int) -> dict | None:
            try:
                r = await client.get(f"{HN_BASE}/item/{sid}.json")
                return r.json() if r.status_code == 200 else None
            except Exception:
                return None

        stories = await asyncio.gather(*[fetch_item(sid) for sid in story_ids])

    results = []
    for s in stories:
        if not s or s.get("type") != "story" or not s.get("title"):
            continue
        title_lower = s["title"].lower()
        if any(kw in title_lower for kw in HN_KEYWORDS):
            results.append({
                "title": s["title"],
                "url": s.get("url", f"https://news.ycombinator.com/item?id={s['id']}"),
                "description": f"HN score: {s.get('score', 0)} — {s.get('descendants', 0)} comments",
            })
        if len(results) >= 10:
            break
    return results[:10]


async def scan_with_claude() -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None
    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        "Search the web for the top 5 news items from this week in data engineering, AI, or MLOps. "
        "Return a JSON object with this exact structure (no markdown, raw JSON only):\n"
        '{"news": [{"title": "...", "source": "...", "score": 8, "tags": ["dbt", "LLM"], '
        '"tech_zoom": "...", "why": "..."}]}\n'
        "score 1-10: rate PODCAST VALUE specifically — does it spark debate? can we teach a concept from it? will practitioners change how they work? is it timely this week? "
        "tech_zoom: 1-sentence technical focus of the item. "
        "why: 1 sentence on what makes this a strong PODCAST SEGMENT — name the angle: a debate to frame, a concept to teach, a surprising practitioner shift, or a hot take worth unpacking."
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{ANTHROPIC_BASE}/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code in (401, 429):
            return None
        resp.raise_for_status()
        data = resp.json()
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block["text"].strip()
                # Strip possible markdown code fences
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = json.loads(text)
                parsed["model_used"] = "claude"
                return parsed
    return None


async def scan_with_mistral() -> dict:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=503, detail="No AI API available")
    news_context = await fetch_hn_news()
    context_text = "\n".join(
        f"- {item['title']} ({item['url']}): {item['description']}" for item in news_context
    ) or "No results from Hacker News."
    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        f"Here are recent news snippets from Hacker News:\n{context_text}\n\n"
        "Based on these, return the top 5 items as a JSON object (no markdown, raw JSON only):\n"
        '{"news": [{"title": "...", "source": "...", "score": 8, "tags": ["dbt", "LLM"], '
        '"tech_zoom": "...", "why": "..."}]}\n'
        "score 1-10: rate PODCAST VALUE specifically — does it spark debate? can we teach a concept from it? will practitioners change how they work? is it timely? "
        "tech_zoom: 1-sentence technical focus of the item. "
        "why: 1 sentence on what makes this a strong PODCAST SEGMENT — name the angle: a debate to frame, a concept to teach, a surprising practitioner shift, or a hot take worth unpacking."
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MISTRAL_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        parsed["model_used"] = "mistral"
        return parsed


async def brief_with_claude(req: BriefRequest) -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None
    prompt = (
        f"You are writing a segment brief for a French-language technical podcast on data/AI/MLOps.\n"
        f"News item: {req.title} (source: {req.source})\n"
        f"Tags: {', '.join(req.tags)}\n"
        f"Why it matters: {req.why}\n\n"
        "Return a JSON object (no markdown, raw JSON only) with this exact structure:\n"
        '{"hook": "...", "news_summary": "...", "practitioner_angle": "...", '
        '"tech_zoom": {"needed": true, "concept": "...", "explanation": "...", "key_tradeoff": "..."}, '
        '"talking_points": ["...", "...", "..."], "closing_question": "..."}'
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{ANTHROPIC_BASE}/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code in (401, 429):
            return None
        resp.raise_for_status()
        data = resp.json()
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block["text"].strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = json.loads(text)
                parsed["model_used"] = "claude"
                return parsed
    return None


async def brief_with_mistral(req: BriefRequest) -> dict:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=503, detail="No AI API available")
    prompt = (
        f"You are writing a segment brief for a French-language technical podcast on data/AI/MLOps.\n"
        f"News item: {req.title} (source: {req.source})\n"
        f"Tags: {', '.join(req.tags)}\n"
        f"Why it matters: {req.why}\n\n"
        "Return a JSON object (no markdown, raw JSON only) with this exact structure:\n"
        '{"hook": "...", "news_summary": "...", "practitioner_angle": "...", '
        '"tech_zoom": {"needed": true, "concept": "...", "explanation": "...", "key_tradeoff": "..."}, '
        '"talking_points": ["...", "...", "..."], "closing_question": "..."}'
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MISTRAL_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        parsed["model_used"] = "mistral"
        return parsed


@app.post("/api/scan")
async def scan():
    try:
        result = await scan_with_claude()
        if result:
            return result
    except Exception:
        pass
    return await scan_with_mistral()


@app.post("/api/brief")
async def brief(req: BriefRequest):
    try:
        result = await brief_with_claude(req)
        if result:
            return result
    except Exception:
        pass
    return await brief_with_mistral(req)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
