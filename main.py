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
ARXIV_API = "https://export.arxiv.org/api/query"
REDDIT_SUBS = ["MachineLearning", "dataengineering", "mlops"]
RSS_FEEDS = [
    # --- Data Engineering ---
    ("dbt Blog",            "https://www.getdbt.com/blog/rss"),
    ("Airbyte Blog",        "https://airbyte.com/blog/rss.xml"),
    ("Databricks Blog",     "https://www.databricks.com/feed"),
    ("InfoQ Data/Eng",      "https://feed.infoq.com/ai-ml-data-eng/"),
    ("Towards Data Science","https://towardsdatascience.com/feed"),
    # --- MLOps ---
    ("Weights & Biases",    "https://wandb.ai/fully-connected/rss.xml"),
    ("Evidently AI",        "https://www.evidentlyai.com/blog/rss"),
    ("neptune.ai Blog",     "https://neptune.ai/blog/feed"),
    # --- AI/LLM (limité) ---
    ("The Batch",           "https://www.deeplearning.ai/the-batch/feed/"),
    ("Hugging Face Blog",   "https://huggingface.co/blog/feed.xml"),
    ("OpenAI Blog",         "https://openai.com/blog/rss.xml"),
]

HN_KEYWORDS = {
    "data", "ml", "ai", "llm", "mlops", "dbt", "spark", "kafka", "flink",
    "airflow", "pipeline", "warehouse", "lakehouse", "vector", "embedding",
    "model", "training", "inference", "gpu", "transformer", "rag",
    "openai", "anthropic", "mistral", "gemini", "huggingface",
    "databricks", "snowflake", "duckdb", "polars", "pandas", "python",
}

GITHUB_TOPICS = [
    # Data Engineering
    "data-engineering", "dbt", "data-pipeline", "apache-spark",
    # MLOps
    "mlops", "model-monitoring", "feature-store",
    # AI/LLM (volontairement limité)
    "llm", "rag",
]

MIN_SCORE = 8  # only genuinely surprising topics pass

PODCAST_SCOPE = (
    "TONE: this podcast sounds like a conversation between two data engineers at a bar — "
    "casual, curious, a bit excited. The energy is 't'as vu ce truc ? c'est dingue parce que…' "
    "NOT a press release, NOT a lecture. The listener should feel like they're hearing something cool "
    "from a friend who just found it.\n\n"
    "BAR TEST — ask yourself: would a data engineer genuinely bring this up with a colleague over a drink? "
    "Would they say 'attends, t'as vu ce que X a sorti ?' If yes: good topic. "
    "If it sounds like something from a corporate newsletter: skip.\n\n"
    "IDEAL topics:\n"
    "  - An emerging pattern spotted across multiple independent repos "
    "(e.g. Rust showing up everywhere for agent management — why is that?)\n"
    "  - A new capability that introduces a new architectural pattern you can actually show "
    "(e.g. Claude dynamic workflows — changes how you structure agents, you can demo it)\n"
    "  - A tool or trick that makes you go 'wait, this is way simpler than I thought'\n"
    "  - A surprising benchmark or result that challenges what everyone assumed\n\n"
    "DISTINCTION for big-company releases: ONLY if it introduces a genuinely new technical concept "
    "worth explaining (new pattern, new capability, new architecture). "
    "SKIP if it's funding news, market share, or a minor model update with no architectural novelty.\n\n"
    "THREE equal pillars:\n"
    "  1. Data Engineering (dbt, Spark, Kafka, DuckDB, orchestration, warehouses…)\n"
    "  2. MLOps (deployment, monitoring, feature stores, model lifecycle, observability…)\n"
    "  3. AI/LLM (only architectural patterns and capabilities — not every release)\n"
    "SKIP: opinion debates, funding news, benchmark wars, PR fluff, anything already discussed for weeks."
)

SCORE_RUBRIC = (
    "YOUR DEFAULT ANSWER IS AN EMPTY LIST. Only add an item if you would genuinely "
    "text it to a data engineer friend right now saying 'attends t'as vu ça ?'\n\n"
    "Score with the BAR TEST — not 'is this relevant?' but 'is this genuinely surprising?'\n"
    "  9-10 = you'd stop mid-sentence to show your phone screen to the person next to you\n"
    "   8   = you'd bring it up naturally, it's fresh and concrete enough to explain in 2 minutes\n"
    "   7   = mildly interesting but you've seen similar things before → score 7, do NOT include\n"
    "  1-6  = LinkedIn post, press release, already everywhere, nothing new → do NOT include\n\n"
    "CRITICAL: most weeks, the correct answer is 0 items or 1-2 items. "
    "Returning 4-5 items means you lowered your standards. "
    "A score of 8 must mean genuinely surprising — not just 'relevant to data engineers'."
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


async def fetch_arxiv_papers() -> list[dict]:
    import xml.etree.ElementTree as ET
    since = (date.today() - timedelta(days=7)).strftime("%Y%m%d")
    query = "cat:cs.LG OR cat:cs.AI OR cat:cs.DB"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                ARXIV_API,
                params={
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": 8,
                },
            )
            resp.raise_for_status()
        ns = "http://www.w3.org/2005/Atom"
        root = ET.fromstring(resp.text)
        results = []
        for entry in root.findall(f"{{{ns}}}entry"):
            title = entry.findtext(f"{{{ns}}}title", "").strip().replace("\n", " ")
            summary = entry.findtext(f"{{{ns}}}summary", "").strip()[:200].replace("\n", " ")
            url = entry.findtext(f"{{{ns}}}id", "").strip()
            authors = [a.findtext(f"{{{ns}}}name", "") for a in entry.findall(f"{{{ns}}}author")]
            results.append({
                "source": "ArXiv",
                "title": title,
                "url": url,
                "description": f"{summary}… — {', '.join(authors[:2])}",
            })
        return results
    except Exception:
        return []


async def fetch_rss_feeds() -> list[dict]:
    import xml.etree.ElementTree as ET
    results = []
    cutoff = date.today() - timedelta(days=7)

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for source_name, url in RSS_FEEDS:
            try:
                resp = await client.get(url, headers={"User-Agent": "podcast-brief-bot/1.0"})
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.text)
                # Handle both RSS <item> and Atom <entry>
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//item") or root.findall(".//atom:entry", ns)
                for item in items[:3]:
                    title = (
                        item.findtext("title")
                        or item.findtext("atom:title", namespaces=ns)
                        or ""
                    ).strip()
                    link = (
                        item.findtext("link")
                        or (item.find("atom:link", ns).get("href") if item.find("atom:link", ns) is not None else "")
                        or ""
                    ).strip()
                    desc = (
                        item.findtext("description")
                        or item.findtext("atom:summary", namespaces=ns)
                        or ""
                    ).strip()[:180].replace("\n", " ")
                    if title and link:
                        results.append({
                            "source": source_name,
                            "title": title,
                            "url": link,
                            "description": desc,
                        })
            except Exception:
                continue
    return results


async def fetch_reddit_posts() -> list[dict]:
    results = []
    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "podcast-brief-bot/1.0"},
    ) as client:
        for sub in REDDIT_SUBS:
            try:
                resp = await client.get(
                    f"https://www.reddit.com/r/{sub}/hot.json",
                    params={"limit": 5},
                )
                if resp.status_code != 200:
                    continue
                for post in resp.json().get("data", {}).get("children", []):
                    d = post["data"]
                    if d.get("stickied") or d.get("is_self") and len(d.get("selftext", "")) < 100:
                        continue
                    results.append({
                        "source": f"Reddit r/{sub}",
                        "title": d.get("title", ""),
                        "url": d.get("url", ""),
                        "description": f"↑ {d.get('score', 0)} — {d.get('num_comments', 0)} comments",
                    })
            except Exception:
                continue
    return results


async def scan_with_claude() -> dict | None:
    if not ANTHROPIC_API_KEY:
        return None
    gh_items, arxiv_items, reddit_items, rss_items = await asyncio.gather(
        fetch_github_trending(), fetch_arxiv_papers(), fetch_reddit_posts(), fetch_rss_feeds()
    )

    def fmt_section(label: str, items: list[dict]) -> str:
        if not items:
            return ""
        lines = "\n".join(f"- [{i['source']}] {i['title']} ({i['url']}): {i['description']}" for i in items)
        return f"\n## {label}\n{lines}\n"

    extra_context = (
        fmt_section("Industry News (TechCrunch, VentureBeat, InfoQ, OpenAI, HuggingFace…)", rss_items)
        + fmt_section("GitHub Trending (new tools & practices)", gh_items)
        + fmt_section("ArXiv Papers (recent research)", arxiv_items)
        + fmt_section("Reddit (community discussion)", reddit_items)
    )

    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        f"{PODCAST_SCOPE}\n"
        "Search the web for the latest news this week. "
        f"Also consider these pre-fetched sources:{extra_context}\n"
        "Select the best items across all sources (web search + GitHub + ArXiv + Reddit + RSS blogs). "
        f"{SCORE_RUBRIC}\n"
        "Return ONLY items scoring 8 or above. Most weeks you should return 0-2 items. "
        "Returning more than 3 means you inflated scores — revise down. "
        "If nothing is genuinely surprising this week, return {\"news\": []} — that is the expected and correct answer. "
        "Return a JSON object with this exact structure (no markdown, raw JSON only):\n"
        '{"news": [{"title": "...", "source": "...", "score": 8, "tags": ["dbt", "LLM"], '
        '"tech_zoom": "...", "why": "..."}]}\n'
        "source: exact source name (e.g. 'GitHub', 'ArXiv', 'Reddit r/MachineLearning', publication name). "
        "tech_zoom: 1-sentence technical focus. "
        "why: 1 sentence on WHAT THE LISTENER WILL LEARN — name the specific concept, method, or practice they can take away and apply. Not a debate framing."
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
                "max_tokens": 2500,
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
    gh_items, arxiv_items, reddit_items, rss_items, hn_items = await asyncio.gather(
        fetch_github_trending(), fetch_arxiv_papers(), fetch_reddit_posts(), fetch_rss_feeds(), fetch_hn_news()
    )

    def fmt_section(label: str, items: list[dict]) -> str:
        if not items:
            return ""
        lines = "\n".join(
            f"- [{i['source']}] {i['title']} ({i['url']}): {i['description']}"
            for i in items
        )
        return f"## {label}\n{lines}\n\n"

    context_text = (
        fmt_section("Industry News (TechCrunch, VentureBeat, InfoQ, OpenAI, HuggingFace…)", rss_items)
        + fmt_section("GitHub Trending (new tools & practices)", gh_items)
        + fmt_section("ArXiv Papers (recent research)", arxiv_items)
        + fmt_section("Reddit (community discussion)", reddit_items)
        + fmt_section("Hacker News (secondary — high bar)", hn_items)
    ) or "No external results available."

    prompt = (
        "You are a researcher for a French-language data/AI/MLOps technical podcast targeting senior data engineers and ML practitioners. "
        f"{PODCAST_SCOPE}\n"
        f"Here are recent items from multiple sources:\n\n{context_text}"
        "Select the best items across all sources, respecting the pillar balance above. "
        f"{SCORE_RUBRIC}\n"
        "Return ONLY items scoring 8 or above. Most weeks you should return 0-2 items. "
        "Returning more than 3 means you inflated scores — revise down. "
        "If nothing is genuinely surprising this week, return {\"news\": []} — that is the expected and correct answer. "
        "Return a JSON object (no markdown, raw JSON only):\n"
        '{"news": [{"title": "...", "source": "...", "score": 8, "tags": ["dbt", "LLM"], '
        '"tech_zoom": "...", "why": "..."}]}\n'
        "source: exact source name (e.g. 'GitHub', 'ArXiv', 'Reddit r/MachineLearning'). "
        "tech_zoom: 1-sentence technical focus. "
        "why: 1 sentence on WHAT THE LISTENER WILL LEARN — name the specific concept, method, or practice they can take away and apply. Not a debate framing."
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MISTRAL_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2500,
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
        f"You are writing a podcast segment brief in French for a data/AI/MLOps technical podcast.\n"
        f"TONE: like explaining something cool to a friend at a bar. Casual, direct, no filler. "
        f"Not 'Dans cet épisode nous allons explorer…' but more like 'En gros t'as un truc qui fait X et c'est ouf parce que…'. "
        f"Don't over-explain, don't be formal. Keep it tight.\n\n"
        f"Topic: {req.title} (source: {req.source})\n"
        f"Tags: {', '.join(req.tags)}\n"
        f"Why it's interesting: {req.why}\n\n"
        "Return a JSON object (no markdown, raw JSON only) with this exact structure:\n"
        '{{"hook": "...", "news_summary": "...", "practitioner_angle": "...", '
        '"tech_zoom": {{"needed": true, "concept": "...", "explanation": "...", "key_tradeoff": "..."}}, '
        '"talking_points": ["...", "...", "..."], "closing_question": "...", '
        '"mini_project": {{"title": "...", "goal": "...", "steps": ["...", "...", "...", "..."]}}}}\n\n'
        "hook: 1-2 sentences max, the way you'd open this topic with a friend — punchy and concrete.\n"
        "news_summary: 2-3 sentences explaining what happened, plain language.\n"
        "practitioner_angle: 1-2 sentences on why a data engineer or ML practitioner cares.\n"
        "tech_zoom.concept: the core technical concept in plain words.\n"
        "tech_zoom.explanation: explain the concept clearly — this is the pedagogical part, keep it simple but precise.\n"
        "tech_zoom.key_tradeoff: 1 sentence on the main tradeoff or gotcha to know.\n"
        "talking_points: 3 natural talking points, like you'd walk through it with a colleague — what it is, how it works, real example.\n"
        "closing_question: 1 casual question you'd throw to the listener, like 'et vous vous l'utiliseriez comment ?'\n"
        "mini_project: a small hands-on project (30min–2h) to actually try the concept. "
        "title: short project name. goal: 1 sentence on what you'll prove. "
        "steps: 3-5 concrete steps with specific tools/commands/code snippets where relevant — no vague instructions."
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
                "max_tokens": 2500,
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
        f"You are writing a podcast segment brief in French for a data/AI/MLOps technical podcast.\n"
        f"TONE: like explaining something cool to a friend at a bar. Casual, direct, no filler. "
        f"Not 'Dans cet épisode nous allons explorer…' but more like 'En gros t'as un truc qui fait X et c'est ouf parce que…'. "
        f"Don't over-explain, don't be formal. Keep it tight.\n\n"
        f"Topic: {req.title} (source: {req.source})\n"
        f"Tags: {', '.join(req.tags)}\n"
        f"Why it's interesting: {req.why}\n\n"
        "Return a JSON object (no markdown, raw JSON only) with this exact structure:\n"
        '{{"hook": "...", "news_summary": "...", "practitioner_angle": "...", '
        '"tech_zoom": {{"needed": true, "concept": "...", "explanation": "...", "key_tradeoff": "..."}}, '
        '"talking_points": ["...", "...", "..."], "closing_question": "...", '
        '"mini_project": {{"title": "...", "goal": "...", "steps": ["...", "...", "...", "..."]}}}}\n\n'
        "hook: 1-2 sentences max, the way you'd open this topic with a friend — punchy and concrete.\n"
        "news_summary: 2-3 sentences explaining what happened, plain language.\n"
        "practitioner_angle: 1-2 sentences on why a data engineer or ML practitioner cares.\n"
        "tech_zoom.concept: the core technical concept in plain words.\n"
        "tech_zoom.explanation: explain the concept clearly — this is the pedagogical part, keep it simple but precise.\n"
        "tech_zoom.key_tradeoff: 1 sentence on the main tradeoff or gotcha to know.\n"
        "talking_points: 3 natural talking points, like you'd walk through it with a colleague — what it is, how it works, real example.\n"
        "closing_question: 1 casual question you'd throw to the listener, like 'et vous vous l'utiliseriez comment ?'\n"
        "mini_project: a small hands-on project (30min–2h) to actually try the concept. "
        "title: short project name. goal: 1 sentence on what you'll prove. "
        "steps: 3-5 concrete steps with specific tools/commands/code snippets where relevant — no vague instructions."
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MISTRAL_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2500,
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
