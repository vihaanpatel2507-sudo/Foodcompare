"""
FoodCompare FastAPI Backend
- SerpAPI: Google search results
- Groq: AI recommendation + restaurant extraction from search results

Run: python -m uvicorn backend_main:app --reload --port 8000
Install: pip install fastapi uvicorn httpx groq
"""

import asyncio, json, re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from groq import Groq
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SERP_API_KEY = "2f6d6ff15b3326ec8eab7b76c17094bd17d1826e0a806dc2fc75cfa5000094c8"
GROQ_API_KEY = "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"

groq_client = Groq(api_key=GROQ_API_KEY)



class CompareRequest(BaseModel):
    dish: str
    area: str = "Chandkheda"


async def search_serp(client: httpx.AsyncClient, query: str):
    try:
        resp = await client.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": SERP_API_KEY, "engine": "google", "num": 8},
            timeout=15,
        )
        resp.raise_for_status()
        results = []
        for r in resp.json().get("organic_results", [])[:8]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
            })
        return results
    except Exception as e:
        print(f"[SerpAPI error] {e}")
        return []


def extract_restaurants_from_results(dish: str, swiggy_results: list, zomato_results: list) -> list:
    """Use Groq to extract structured restaurant data from search snippets."""
    try:
        swiggy_text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in swiggy_results)
        zomato_text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in zomato_results)

        prompt = f"""Extract restaurant information from these search results for "{dish}" in Chandkheda, Ahmedabad.

SWIGGY RESULTS:
{swiggy_text}

ZOMATO RESULTS:
{zomato_text}

Extract up to 8 restaurants mentioned. For each restaurant return a JSON array with objects having these fields:
- name: restaurant name (string)
- dish_price: price of {dish} if mentioned, else "" (string like "₹150" or "₹150-200")
- cuisine: cuisine type (string)
- platform: "swiggy" or "zomato" or "both" (which platform it appeared on)
- area: area name like "Chandkheda" (string)
- swiggy_url: "https://www.swiggy.com/search?query={dish.replace(' ', '%20')}"
- zomato_url: "https://www.zomato.com/ahmedabad/search?q=" + restaurant name url encoded

Return ONLY a valid JSON array, no explanation, no markdown, no code blocks. Just the raw JSON array."""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        # Clean up any markdown if present
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        restaurants = json.loads(raw)
        print(f"[Groq Extract] Found {len(restaurants)} restaurants")

        # Add map search URL for each
        for r in restaurants:
            name = r.get("name", "")
            r["map_url"] = f"https://www.google.com/maps/search/{name.replace(' ', '+')}+Chandkheda+Ahmedabad"
            r["zomato_url"] = f"https://www.zomato.com/ahmedabad/search?q={name.replace(' ', '%20')}"

        return restaurants

    except Exception as e:
        print(f"[Extract error] {e}")
        return []


@app.post("/compare")
async def compare(req: CompareRequest):
    dish = req.dish.strip()

    async with httpx.AsyncClient() as client:
        serp_swiggy, serp_zomato = await asyncio.gather(
            search_serp(client, f"{dish} swiggy {req.area} ahmedabad price"),
            search_serp(client, f"{dish} zomato {req.area} ahmedabad price"),
        )

    # Build AI recommendation prompt
    swiggy_text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in serp_swiggy) or "No results"
    zomato_text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in serp_zomato) or "No results"

    rec_prompt = f"""You are a food price comparison agent for {req.area}, Ahmedabad.
User searched for: "{dish}"

Swiggy results:
{swiggy_text}

Zomato results:
{zomato_text}

Give:
1. Brief summary of prices on each platform.
2. Clear recommendation: which is better and why.
Be concise and friendly. Use ₹ for prices."""

    loop = asyncio.get_event_loop()

    # Run AI recommendation and restaurant extraction in parallel
    try:
        groq_rec, restaurants = await asyncio.gather(
            loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": rec_prompt}],
                max_tokens=400,
            )),
            loop.run_in_executor(None, lambda: extract_restaurants_from_results(dish, serp_swiggy, serp_zomato)),
        )
        ai_recommendation = groq_rec.choices[0].message.content
    except Exception as e:
        ai_recommendation = f"AI unavailable: {e}"
        restaurants = []

    return {
        "dish": dish,
        "swiggy_search": serp_swiggy,
        "zomato_search": serp_zomato,
        "restaurants": restaurants,
        "ai_recommendation": ai_recommendation,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
