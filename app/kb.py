"""Knowledge base: BM25 text retrieval over scraped pages + structured Wasalt listing search.

Pure-python (no heavy ML deps) so the Docker image stays small and cold-starts fast.
"""
from __future__ import annotations
import json, math, re
from collections import Counter, defaultdict
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with you your we our i me my can do does what which who how when where about tell give show any all".split())


KIND_BOOST = {"about": 1.6, "home": 1.3, "faq": 1.4, "why-invest": 1.3, "projects": 1.3, "partners": 1.2,
              "investor": 1.3, "static": 1.0, "category": 1.1}


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf: list[Counter] = []
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[int]] = defaultdict(list)
        lens = []
        for i, c in enumerate(chunks):
            # title is boosted by repeating it
            toks = tokenize(c["title"] + " " + c["title"] + " " + c["text"])
            tf = Counter(toks)
            self.tf.append(tf)
            lens.append(len(toks))
            for t in tf:
                self.df[t] += 1
                self.postings[t].append(i)
        self.avgdl = (sum(lens) / len(lens)) if lens else 1.0
        self.lens = lens
        self.N = len(chunks)

    def search(self, query: str, k: int = 5, site: str | None = None, kind: str | None = None) -> list[dict]:
        q = tokenize(query)
        if not q:
            return []
        scores: dict[int, float] = defaultdict(float)
        for t in set(q):
            if t not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        # evergreen pages (about/FAQ/projects) beat time-bound press/blog items on generic questions
        for i in list(scores):
            scores[i] *= KIND_BOOST.get(self.chunks[i].get("kind"), 1.0)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out, seen_pages = [], Counter()
        for i, s in ranked:
            c = self.chunks[i]
            if site and site != "any" and c["site"] != site:
                continue
            if kind and c.get("kind") != kind:
                continue
            if seen_pages[c["url"]] >= 2:  # diversify: max 2 chunks per page
                continue
            seen_pages[c["url"]] += 1
            out.append({**c, "score": round(s, 2)})
            if len(out) >= k:
                break
        return out


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        def read_jsonl(p: Path):  # "\n" only: splitlines() would also split on U+2028 inside strings
            return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
        self.chunks = read_jsonl(kb_dir / "chunks.jsonl")
        lp = kb_dir / "listings.jsonl"
        self.listings = read_jsonl(lp) if lp.exists() else []
        mp = kb_dir / "meta.json"
        self.meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        self.index = BM25Index(self.chunks)
        self.pages: dict[str, list[dict]] = defaultdict(list)
        for c in self.chunks:
            self.pages[c["url"]].append(c)

    def search(self, query, k=5, site=None, kind=None):
        return self.index.search(query, k, site, kind)

    def get_page(self, url: str, max_chars: int = 6000):
        cs = self.pages.get(url)
        if not cs:
            # tolerate trailing-slash / www differences
            key = next((u for u in self.pages if u.rstrip("/") == url.rstrip("/")), None)
            cs = self.pages.get(key) if key else None
        if not cs:
            return None
        cs = sorted(cs, key=lambda c: c["chunk"])
        return {"url": cs[0]["url"], "title": cs[0]["title"], "text": "\n".join(c["text"] for c in cs)[:max_chars]}

    def search_listings(self, listing_type=None, property_type=None, city=None, district=None,
                        min_price=None, max_price=None, min_bedrooms=None, limit=6):
        def ok(l):
            if listing_type and l.get("listing_type") != listing_type: return False
            if property_type and property_type.lower() not in (l.get("property_type") or "").lower(): return False
            if city and city.lower() not in (l.get("city") or "").lower() and city.lower() not in (l.get("location") or "").lower(): return False
            if district and district.lower() not in (l.get("location") or "").lower(): return False
            p = l.get("price")
            if min_price is not None and (p is None or p < min_price): return False
            if max_price is not None and (p is None or p > max_price): return False
            if min_bedrooms is not None and (l.get("bedrooms") is None or l["bedrooms"] < min_bedrooms): return False
            return True
        res = [l for l in self.listings if ok(l)]
        res.sort(key=lambda l: (l.get("price") is None, l.get("price") or 0))
        return res[:limit], len(res)
