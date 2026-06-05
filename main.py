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
    "   5-6 = weak — no strong angle, do NOT include\n"
    "   1-4 = skip — too niche, too old, no discussion value\n"
    "Be strict and conservative. If a Hacker News item is just a link dump or minor release, score it low. "
    "Prefer GitHub repos that introduce a genuinely new tool or practice over HN articles about things already well-known. "
    "A slow week means an empty or near-empty list — that is the correct answer. "
    "Never inflate scores to fill slots. An empty list is better than mediocre content."
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
        if len(results) >= 5:
            break
    return results[:5]


async def fetch_github_trending() -> list[dict]:
    since = (date.today() - timedelta(days=14)).isoformat()
    seen_urls: set[str] = set()
    results = []
    async with httpx.AsyncClient(timeout=20) as client:
        for topic in GITHUB_TOPICS[:4]:
            try:
                resp = await client.get(
                    GITHUB_API,
                    headers={"Accept": "application/vnd.github+json"},
                    params={
                        "q": f"topic:{topic} pushed:>{since} stars:>10",
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 4,
                    },
                )
                if resp.status_code != 200:
                    continue
                for repo in resp.json().get("items", []):
                    url = repo["html_url"]
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    desc = repo.get("description") or ""
                    results.append({
                        "source": "GitHub",
                        "title": f"{repo['full_name']} — {desc}",
                        "url": url,
                        "description": (
                            f"★ {repo.get('stargazers_count', 0)} stars — "
                            f"topics: {', '.join(repo.get('topics', [])[:5])}"
                        ),
                    })
            except Exception:
                continue
    return results[:12]


async def scan_with_claude() -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None
    gh_items = await fetch_github_trending()
    gh_context = ""
    if gh_items:
        gh_lines = "\n".join(
            f"- {item['title']} ({item['url']}): {item['description']}"
            for item in gh_items
        )
        gh_context = f"\n\nHere are trending GitHub repos fetched right now — consider them alongside your web search results:\n{gh_lines}"

    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        "Search the web for the latest news in data engineering, AI, or MLOps this week."
        f"{gh_context}\n\n"
        "Combine web search results and the GitHub repos above to select the best items. "
        "A GitHub repo counts as a valid item if it introduces a genuinely new tool or practice. "
        "Hacker News items need a strong discussion angle to qualify — a link to a known library update is not enough. "
        f"{SCORE_RUBRIC}\n"
        "Return ONLY items scoring 7 or above. If nothing clears that bar, return {\"news\": []}. "
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
    if gh_items:
        context_text += f"## GitHub Trending (priority source)\n{fmt(gh_items)}\n\n"
    if hn_items:
        context_text += f"## Hacker News (secondary source — high bar required)\n{fmt(hn_items)}\n"
    if not context_text:
        context_text = "No external results available."

    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        f"Here are recent items — GitHub repos are the priority source, Hacker News is secondary:\n\n{context_text}\n"
        "An HN item must have a strong discussion angle to qualify — a minor release or link dump does not. "
        f"{SCORE_RUBRIC}\n"
        "Return ONLY items scoring 7 or above. If nothing clears that bar, return {\"news\": []}. "
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
