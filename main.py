from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import httpx
import os
import json

app = FastAPI(title="FactCheck API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")  # Optional but improves accuracy


class FactCheckRequest(BaseModel):
    text: str
    username: str = ""
    platform: str = "instagram_reels"


class FactCheckResponse(BaseModel):
    verdict: str        # TRUE | FALSE | MISLEADING | UNVERIFIABLE
    explanation: str
    confidence: str     # HIGH | MEDIUM | LOW
    sources: str = ""


async def web_search(query: str) -> str:
    """Search the web for context about a claim."""
    if not BRAVE_API_KEY:
        return ""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
                params={"q": query, "count": 5},
                timeout=10
            )
            data = response.json()
            results = data.get("web", {}).get("results", [])
            
            snippets = []
            for r in results[:3]:
                snippets.append(f"- {r.get('title', '')}: {r.get('description', '')}")
            
            return "\n".join(snippets)
    except Exception:
        return ""


@app.post("/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest):
    if len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short to fact-check")
    
    # Limit text length
    text = request.text[:2000]
    
    # Search for context
    search_context = await web_search(f"fact check: {text[:200]}")
    
    system_prompt = """You are a professional fact-checker specializing in social media misinformation.

Your job is to analyze text from Instagram Reels and determine if the claims are accurate.

Always respond with a JSON object in this exact format:
{
  "verdict": "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIABLE",
  "explanation": "Clear, concise explanation in 2-3 sentences. Be specific about what is true or false.",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "sources": "Brief mention of what sources would verify this"
}

Guidelines:
- TRUE: Claims are factually accurate and verifiable
- FALSE: Claims contain clear factual errors
- MISLEADING: Contains some truth but framed deceptively, missing key context, or exaggerated
- UNVERIFIABLE: Cannot be verified without more context (opinions, predictions, personal stories)
- Be direct and educational, not preachy
- Focus on the main factual claims, ignore opinions
- If text is entertainment/comedy, say UNVERIFIABLE

Respond ONLY with the JSON object. No other text."""

    user_message = f"""Fact-check this content from Instagram Reels:

CONTENT:
{text}

{f'POSTED BY: @{request.username}' if request.username else ''}

{f'WEB SEARCH CONTEXT:{chr(10)}{search_context}' if search_context else ''}

Analyze and return the JSON verdict."""

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    try:
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        result = json.loads(raw.strip())
        
        return FactCheckResponse(
            verdict=result.get("verdict", "UNVERIFIABLE"),
            explanation=result.get("explanation", "Could not analyze this content."),
            confidence=result.get("confidence", "LOW"),
            sources=result.get("sources", "")
        )
    except json.JSONDecodeError:
        # Fallback — return raw text as explanation
        return FactCheckResponse(
            verdict="UNVERIFIABLE",
            explanation=response.content[0].text[:300],
            confidence="LOW"
        )


@app.get("/health")
def health():
    return {"status": "ok"}