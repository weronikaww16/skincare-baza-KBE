"""
Przegląda kanały z config.json, wysyła nowe filmy do Gemini
i dopisuje wyniki do bazy w docs/data/.
"""
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import unicodedata

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / "docs" / "data"
VIDEOS_F = DATA / "videos.json"
PRODUCTS_F = DATA / "products.json"
KNOW_F = DATA / "knowledge.json"

API_KEY = os.environ.get("GEMINI_API_KEY")
MODELS = CONFIG.get("models") or [
    CONFIG.get("model", "gemini-flash-latest"),
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
]
MODELS = list(dict.fromkeys(MODELS))
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
MAX_ATTEMPTS = 3
BACKLOG_PER_CHANNEL = CONFIG.get("backlog_per_channel", 100)


class RateLimited(Exception):
    pass


# ---------- pliki ----------

def load(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return default


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- lista filmów ----------

def list_videos(handle):
    cmd = [
        "yt-dlp", "--flat-playlist", "--dump-json",
        "--extractor-args", "youtubetab:approximate_date",
        f"https://www.youtube.com/{handle}/videos",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    videos = []
    for line in res.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        date = None
        ts = e.get("timestamp") or e.get("release_timestamp")
        if ts:
            date = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")
        elif e.get("upload_date"):
            u = e["upload_date"]
            date = f"{u[:4]}-{u[4:6]}-{u[6:8]}"
        videos.append({
            "id": e["id"],
            "title": e.get("title", ""),
            "duration": int(e.get("duration") or 1200),
            "date": date,
        })
    if not videos:
        print(f"[!] Nie udało się pobrać listy filmów dla {handle}:\n{res.stderr[-1500:]}")
    dated = sum(1 for v in videos if v["date"])
    print(f"{handle}: {len(videos)} filmów na kanale, z datą: {dated}")
    return videos


# ---------- Gemini ----------

PROMPT_FULL = """Obejrzyj i przesłuchaj cały ten film z kanału o pielęgnacji skóry.
Wypisz WYŁĄCZNIE to, co mówi lub pokazuje autor. Niczego nie dopowiadaj od siebie.

Zwróć JSON o dokładnie takiej strukturze:
{
  "summary": "2-3 zdania, o czym jest film",
  "products": [
    {
      "brand": "marka dokładnie jak w filmie",
      "name": "nazwa produktu",
      "category": "np. krem, serum, tonik, żel do mycia, SPF, maska",
      "verdict": "polecany | odradzany | neutralny",
      "reason": "krótko, dlaczego autor tak uważa (skład, działanie, cena itd.)",
      "ingredients": ["kluczowe składniki, o których mówi autor"],
      "for": ["typy skóry lub problemy, do których autor go przypisuje"],
      "timestamp": "MM:SS"
    }
  ],
  "ingredient_rules": [
    {
      "type": "nie łączyć | można łączyć | uwaga",
      "ingredients": ["składnik A", "składnik B"],
      "rule": "zasada jednym zdaniem",
      "why": "wyjaśnienie autora",
      "timestamp": "MM:SS"
    }
  ],
  "routines": [
    {
      "problem": "np. trądzik, przebarwienia, sucha skóra, rozszerzone pory",
      "time_of_day": "rano | wieczór | cały dzień",
      "steps": ["krok 1", "krok 2"],
      "notes": "dodatkowe uwagi autora",
      "timestamp": "MM:SS"
    }
  ],
  "tips": [
    {"topic": "temat", "tip": "rada autora", "timestamp": "MM:SS"}
  ]
}

Zasady:
- "polecany" = autor chwali produkt lub jego skład; "odradzany" = krytykuje skład lub odradza; "neutralny" = tylko wspomina bez oceny.
- Jeśli czegoś w filmie nie ma, zostaw pustą listę.
- Pisz po polsku. Zwróć tylko JSON."""

PROMPT_PRODUCTS = """Obejrzyj i przesłuchaj cały ten film.
Interesują mnie WYŁĄCZNIE oceny kosmetyków wypowiadane przez autora: które produkty mają dobry skład i są polecane, a które mają zły skład lub są odradzane.
Niczego nie dopowiadaj od siebie.

Zwróć JSON:
{
  "summary": "1-2 zdania, o czym jest film",
  "products": [
    {
      "brand": "marka dokładnie jak w filmie",
      "name": "nazwa produktu",
      "category": "np. podkład, krem, serum, pomadka",
      "verdict": "polecany | odradzany | neutralny",
      "reason": "krótko, co autor mówi o składzie lub działaniu",
      "ingredients": ["składniki, o których wspomina autor"],
      "for": ["typy skóry lub problemy, jeśli autor je podaje"],
      "timestamp": "MM:SS"
    }
  ]
}
Jeśli w filmie nie ma ocen produktów, zwróć pustą listę "products". Pisz po polsku. Zwróć tylko JSON."""


active_model = 0


def ask_gemini(video_id, prompt):
    global active_model
    body = {
        "contents": [{
            "parts": [
                {"file_data": {"file_uri": f"https://www.youtube.com/watch?v={video_id}"}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "mediaResolution": "MEDIA_RESOLUTION_LOW",
            "temperature": 0.2,
        },
    }
    waits = 0
    while active_model < len(MODELS):
        model = MODELS[active_model]
        r = requests.post(BASE_URL.format(model), headers={"x-goog-api-key": API_KEY}, json=body, timeout=900)
        if r.status_code == 200:
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            return json.loads(text)
        msg = r.text[:800]
        if r.status_code in (404, 400) and "model" in msg.lower() and "not" in msg.lower():
            print(f"   model {model} niedostępny, próbuję następny")
            active_model += 1
            continue
        if r.status_code == 429:
            daily = "perday" in msg.lower().replace("_", "").replace(" ", "") or "limit: 0" in msg
            print(f"   [{model}] odmowa 429: {msg}")
            if daily or waits >= 2:
                print(f"   przełączam z modelu {model} na następny")
                active_model += 1
                waits = 0
                continue
            waits += 1
            time.sleep(70)
            continue
        if r.status_code >= 500:
            time.sleep(30)
            continue
        raise RuntimeError(f"Gemini {r.status_code}: {msg}")
    raise RateLimited("Wszystkie modele wyczerpały dzienny limit")


# ---------- scalanie ----------

def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def merge_products(products, found, source):
    index = {p["key"]: p for p in products}
    for f in found:
        name = (f.get("name") or "").strip()
        brand = (f.get("brand") or "").strip()
        if not name:
            continue
        key = norm(f"{brand} {name}")
        verdict = (f.get("verdict") or "neutralny").strip().lower()
        mention = {
            "verdict": verdict,
            "reason": f.get("reason", ""),
            **source,
            "timestamp": f.get("timestamp", ""),
        }
        p = index.get(key)
        if not p:
            p = {"key": key, "brand": brand, "name": name, "category": f.get("category", ""),
                 "ingredients": [], "for": [], "mentions": []}
            products.append(p)
            index[key] = p
        if any(m["video_id"] == source["video_id"] for m in p["mentions"]):
            continue
        p["mentions"].append(mention)
        p["ingredients"] = sorted(set(p["ingredients"]) | set(f.get("ingredients") or []))
        p["for"] = sorted(set(p["for"]) | set(f.get("for") or []))
    for p in products:
        rated = [m for m in p["mentions"] if m["verdict"] in ("polecany", "odradzany")]
        rated.sort(key=lambda m: m.get("date") or "")
        p["verdict"] = rated[-1]["verdict"] if rated else "neutralny"
        p["conflict"] = len({m["verdict"] for m in rated}) > 1
    products.sort(key=lambda p: (p["brand"].lower(), p["name"].lower()))
    return products


def merge_knowledge(know, result, source):
    for field in ("ingredient_rules", "routines", "tips"):
        for item in result.get(field) or []:
            item.update({k: v for k, v in source.items()})
            know.setdefault(field, []).append(item)
    return know


# ---------- główna pętla ----------

def main():
    if not API_KEY:
        sys.exit("Brak GEMINI_API_KEY")

    videos = load(VIDEOS_F, {})
    products = load(PRODUCTS_F, [])
    know = load(KNOW_F, {"ingredient_rules": [], "routines": [], "tips": []})
    since = CONFIG.get("since", "2000-01-01")

    queue = []
    for ch in CONFIG["channels"]:
        vids = list_videos(ch["handle"])
        if not any(v["date"] for v in vids):
            vids = vids[:BACKLOG_PER_CHANNEL]
            print(f"   brak dat, biorę {len(vids)} najnowszych filmów z kanału")
        for rank, v in enumerate(vids):
            v["rank"] = rank
            state = videos.get(v["id"], {})
            if state.get("status") == "done" or state.get("attempts", 0) >= MAX_ATTEMPTS:
                continue
            if v["date"] and v["date"] < since:
                continue
            queue.append({**v, "channel": ch["name"], "mode": ch["mode"]})

    # najnowsze najpierw, kanały na przemian
    queue.sort(key=lambda v: (v["date"] or "9999", -v["rank"]), reverse=True)
    budget = CONFIG.get("daily_budget_minutes", 420) * 60
    limit = CONFIG.get("max_videos_per_run", 40)
    print(f"Do obejrzenia: {len(queue)} filmów")

    used, done = 0, 0
    for v in queue:
        if done >= limit or used + v["duration"] > budget:
            break
        print(f"-> {v['channel']}: {v['title']} ({v['duration'] // 60} min, {v['date'] or 'brak daty'})")
        prompt = PROMPT_FULL if v["mode"] == "full" else PROMPT_PRODUCTS
        source = {"video_id": v["id"], "video_title": v["title"], "channel": v["channel"], "date": v["date"]}
        try:
            result = ask_gemini(v["id"], prompt)
        except RateLimited:
            print("Limit dzienny wyczerpany, reszta jutro.")
            break
        except Exception as e:
            st = videos.setdefault(v["id"], {"title": v["title"], "channel": v["channel"], "date": v["date"]})
            st["attempts"] = st.get("attempts", 0) + 1
            st["status"] = "failed"
            st["error"] = str(e)[:300]
            print(f"   błąd: {e}")
            save(VIDEOS_F, videos)
            continue

        products = merge_products(products, result.get("products") or [], source)
        if v["mode"] == "full":
            know = merge_knowledge(know, result, source)
        videos[v["id"]] = {"title": v["title"], "channel": v["channel"], "date": v["date"],
                           "status": "done", "summary": result.get("summary", "")}
        used += v["duration"]
        done += 1
        save(PRODUCTS_F, products)
        save(KNOW_F, know)
        save(VIDEOS_F, videos)
        print(f"   ok ({MODELS[active_model]}), produktów w filmie: {len(result.get('products') or [])}")
        time.sleep(20)

    print(f"Gotowe: {done} filmów, łącznie produktów w bazie: {len(products)}")


if __name__ == "__main__":
    main()
