"""Turn raw crawl output (data/raw/*.jsonl) into the retrieval knowledge base (data/kb/).

Outputs
  chunks.jsonl    - ~900-char text chunks (title/url/site/kind) for BM25 retrieval
  listings.jsonl  - structured Wasalt listings (price, bedrooms, area, city...) for filtered search
  meta.json       - snapshot date + corpus stats
"""
import hashlib, json, re, sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "kb"
OUT.mkdir(parents=True, exist_ok=True)
CHUNK = 900


def load(name):
    p = RAW / name
    if not p.exists():
        return []
    # split on "\n" only: str.splitlines() also breaks on U+2028 which can appear inside scraped strings
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
    return [r for r in rows if "url" in r and "text" in r]


def chunk_text(text: str):
    buf, out = "", []
    for line in [l.strip() for l in text.split("\n") if l.strip()]:
        if len(buf) + len(line) > CHUNK and buf:
            out.append(buf)
            buf = buf[-120:] if len(buf) > 120 else ""  # small overlap
        buf += ("\n" if buf else "") + line
    if buf.strip():
        out.append(buf)
    return out


def clean_title(t: str) -> str:
    return re.sub(r"\s*[|\-–]\s*(Dar ?Global|Wasalt).*$", "", t or "", flags=re.I).strip() or t


def num(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


# ---------- Wasalt ----------
def parse_listing(r):
    t, url = r["text"], r["url"]
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    m = re.search(r"/property/(sale|rent)/", url)
    ltype = m.group(1) if m else None
    title = clean_title(r["title"]).replace(" For Sale", "").replace(" For Rent", "")
    # location block: lines before "Map view"
    loc = ""
    if "Map view" in lines:
        i = lines.index("Map view")
        loc = ", ".join(l.strip(", ") for l in lines[max(0, i - 6):i] if l.strip(", ") and l not in (",",) and l != title and not l.startswith(("For ", "Semi", "Furnished", "Unfurnished")) and "Apartment" != l)
        loc = re.sub(r"\s*,\s*,", ",", loc)
        after = lines[i + 1:]
    else:
        after = lines
    price = None
    for l in after:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", l):
            price = num(l)
            break
    if price is None:
        pm = re.search(r"Price:?\s*([\d,]+)\s*SAR", t)
        price = num(pm.group(1)) if pm else None
    bm = re.search(r"(\d+)\s*Bedrooms?", title) or re.search(r"\n(\d+)\n\s*Bedrooms?", t)
    am = re.search(r"([\d,.]+)\s*\n?\s*SQM", t) or re.search(r"([\d,.]+)\s*sqm", t, re.I)
    ptype = next((k for k in ("villa", "apartment", "land", "building", "floor", "rest house", "shop", "office", "warehouse", "farm", "townhouse", "duplex", "studio") if k in title.lower()), "other")
    city = loc.split(",")[-1].strip() if loc else ""
    desc = (r.get("desc") or "")[:200]
    return {"url": "https://wasalt.sa" + url if url.startswith("/") else url, "title": title, "listing_type": ltype,
            "property_type": ptype, "price": price, "bedrooms": int(bm.group(1)) if bm else None,
            "area_sqm": num(am.group(1)) if am else None, "location": loc, "city": city, "summary": desc}


def wasalt_docs(rows):
    docs, listings, seen = [], [], set()
    for r in rows:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if r["kind"] == "listing":
            l = parse_listing(r)
            if l["price"] is not None and l["location"]:  # drop records whose page body was truncated in the crawl
                listings.append(l)
            continue
        if h in seen or len(r["text"]) < 150:
            continue  # SPA fallbacks return the homepage for unknown paths
        seen.add(h)
        url = "https://wasalt.sa" + r["url"]
        if r["kind"] == "category":
            n = re.search(r"([\d,]+)\s*results", r["text"])
            parts = [r["desc"], f"{r['h1'] or r['title']}: Wasalt shows {n.group(1)} listings for this search at snapshot time." if n else ""]
            for f in r.get("faq", []):
                parts.append(f"Q: {f['q']}\nA: {f['a']}")
            text = "\n".join(p for p in parts if p)
        else:
            text = r["text"]
        if r["kind"] == "category":
            title = (r["h1"] or clean_title(r["title"])).strip()
        else:  # static pages share a generic <title>; derive a readable one from the slug
            slug = r["url"].rstrip("/").split("/")[-1]
            title = "Wasalt " + (slug.replace("-n-", " and ").replace("-", " ").title() if slug not in ("en", "") else "Home")
        docs.append({"site": "wasalt", "kind": r["kind"], "url": url, "title": title, "text": text})
    return docs, listings


# ---------- Dar Global ----------
def darglobal_docs(rows):
    docs, seen = [], set()
    for r in rows:
        if len(r["text"]) < 200:
            continue
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        text = (r.get("desc", "") + "\n" if r.get("desc") else "") + r["text"]
        if r.get("date"):
            text = f"Published: {r['date'][:10]}\n" + text
        docs.append({"site": "darglobal", "kind": r.get("kind", "page"), "url": r["url"],
                     "title": clean_title(r["title"]) or r.get("h1", ""), "text": text})
    return docs


def main():
    dg = darglobal_docs(load("darglobal.jsonl"))
    wd, listings = wasalt_docs(load("wasalt.jsonl"))
    chunks = []
    for d in dg + wd:
        for i, c in enumerate(chunk_text(d["text"])):
            chunks.append({"site": d["site"], "kind": d["kind"], "url": d["url"], "title": d["title"], "chunk": i, "text": c})
    (OUT / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    (OUT / "listings.jsonl").write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in listings), encoding="utf-8")
    meta = {"scraped_at": str(date.today()), "darglobal_pages": len(dg), "wasalt_pages": len(wd),
            "wasalt_listings": len(listings), "chunks": len(chunks)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)
    print(Counter(c["kind"] for c in chunks).most_common(12))


if __name__ == "__main__":
    sys.exit(main())
