import os
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import date, timedelta
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
GITHUB_API = "https://api.github.com/search/repositories"

HN_KEYWORDS = {
    "data", "ml", "ai", "llm", "mlops", "dbt", "spark", "kafka", "flink",
    "airflow", "pipeline", "warehouse", "lakehouse", "vector", "embedding",
    "model", "training", "inference", "gpu", "transformer", "rag",
    "openai", "anthropic", "mistral", "gemini", "huggingface",
    "databricks", "snowflake", "duckdb", "polars", "pandas", "python",
}

GITHUB_TOPICS = [
    "llm", "mlops", "data-engineering", "machine-learning",
    "rag", "vector-database", "dbt", "data-pipeline",
]

MIN_SCORE = 6  # items below this score are dropped — not worth a segment

SCORE_RUBRIC = (
    "Score the PODCAST VALUE on a strict 1-10 scale:\n"
    "  9-10 = unmissable — strong debate OR teaches a key concept, highly timely, practitioners will talk about it\n"
    "   7-8 = solid segment — clear angle, practitioners care, worth 10 minutes\n"
    "   5-6 = weak — possible filler, no strong angle\n"
    "   1-4 = skip — too niche, too old, no discussion value\n"
    "Be strict: a slow news week should produce low scores, not inflated ones. "
    "Never give 9+ just because something is recent. Never pad to reach 5 items."
)


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
                "source": "Hacker News",
                "title": s["title"],
                "url": s.get("url", f"https://news.ycombinator.com/item?id={s['id']}"),
                "description": f"HN score: {s.get('score', 0)} — {s.get('descendants', 0)} comments",
            })
        if len(results) >= 10:
            break
    return results[:10]


async def fetch_github_trending() -> list[dict]:
    since = (date.today() - timedelta(days=14)).isoformat()
    results = []
    async with httpx.AsyncClient(timeout=20) as client:
        for topic in GITHUB_TOPICS[:4]:
            try:
                resp = await client.get(
                    GITHUB_API,
                    headers={"Accept": "application/vnd.github+json"},
                    params={
                        "q": f"topic:{topic} pushed:>{since} stars:>50",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 3,
                    },
                )
                if resp.status_code != 200:
                    continue
                for repo in resp.json().get("items", []):
                    results.append({
                        "source": "GitHub",
                        "title": f"{repo['full_name']} — {repo.get('description', '')}",
                        "url": repo["html_url"],
                        "description": (
                            f"★ {repo.get('stargazers_count', 0)} stars "
                            f"({repo.get('stargazers_count', 0) - repo.get('watchers_count', 0):+d} recent) — "
                            f"topics: {', '.join(repo.get('topics', [])[:5])}"
                        ),
                    })
            except Exception:
                continue
    return results[:8]


async def scan_with_claude() -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None
    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        "Search the web AND GitHub for items from this week in data engineering, AI, or MLOps. "
        "Include trending GitHub repos if they represent a new tool, pattern, or practice worth discussing. "
        f"{SCORE_RUBRIC}\n"
        "Return only items scoring 7 or above. Return fewer than 5 if the week doesn't have enough strong content — an empty list is valid. "
        "Return a JSON object with this exact structure (no markdown, raw JSON only):\n"
        '{"news": [{"title": "...", "source": "...", "score": 8, "tags": ["dbt", "LLM"], '
        '"tech_zoom": "...", "why": "..."}]}\n'
        "source: publication name or 'GitHub'. "
        "tech_zoom: 1-sentence technical focus. "
        "why: 1 sentence on the PODCAST ANGLE — a debate to frame, a concept to teach, a new practice to explore, or a hot take worth unpacking."
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
    hn_items, gh_items = await asyncio.gather(fetch_hn_news(), fetch_github_trending())

    def fmt(items: list[dict]) -> str:
        return "\n".join(
            f"- [{item['source']}] {item['title']} ({item['url']}): {item['description']}"
            for item in items
        )

    context_text = ""
    if hn_items:
        context_text += f"## Hacker News\n{fmt(hn_items)}\n\n"
    if gh_items:
        context_text += f"## GitHub Trending\n{fmt(gh_items)}\n"
    if not context_text:
        context_text = "No external results available."

    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        f"Here are recent items from Hacker News and GitHub trending repos:\n\n{context_text}\n"
        f"{SCORE_RUBRIC}\n"
        "Return only items scoring 7 or above. Return fewer than 5 if the content doesn't warrant it — an empty list is valid. "
        "Return a JSON object (no markdown, raw JSON only):\n"
        '{"news": [{"title": "...", "source": "...", "score": 8, "tags": ["dbt", "LLM"], '
        '"tech_zoom": "...", "why": "..."}]}\n'
        "source: publication name or 'GitHub'. "
        "tech_zoom: 1-sentence technical focus. "
        "why: 1 sentence on the PODCAST ANGLE — a debate to frame, a concept to teach, a new practice to explore, or a hot take worth unpacking."
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


def filter_news(result: dict) -> dict:
    news = [n for n in result.get("news", []) if n.get("score", 0) >= MIN_SCORE]
    result["news"] = sorted(news, key=lambda n: n.get("score", 0), reverse=True)
    return result


@app.post("/api/scan")
async def scan():
    try:
        result = await scan_with_claude()
        if result:
            return filter_news(result)
    except Exception:
        pass
    return filter_news(await scan_with_mistral())


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
