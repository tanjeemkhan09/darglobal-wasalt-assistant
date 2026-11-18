"""Scrape public Dar Global pages listed in its sitemap (robots.txt-compliant, polite rate)."""
import json, re, sys, time
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (compatible; DarGlobalAssistantResearch/1.0)"}
OUT = "data/raw/darglobal.jsonl"
BLOCKED = re.compile(r"utm_|\?s=|\?post_type=|&&|\?trk=|tp_sp|menu-bwvsd")

def sitemap_urls():
    xml = requests.get("https://darglobal.co.uk/sitemap.xml", headers=UA, timeout=30).text
    return [u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if not BLOCKED.search(u)]

def clean(html, url):
    s = BeautifulSoup(html, "lxml")
    for t in s(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"]):
        t.decompose()
    title = (s.title.string or "").strip() if s.title else ""
    desc = (s.find("meta", attrs={"name": "description"}) or {}).get("content", "")
    h1 = s.find("h1").get_text(" ", strip=True) if s.find("h1") else ""
    main = s.find("main") or s.body
    lines = []
    for el in main.find_all(["h1","h2","h3","h4","p","li","td","th","dt","dd","span","div"]):
        if el.name in ("span", "div") and el.find(["p","li","h1","h2","h3","h4","div"]):
            continue
        t = el.get_text(" ", strip=True)
        if t and (not lines or lines[-1] != t):
            lines.append(t)
    return {"site": "darglobal", "url": url, "title": title, "desc": desc, "h1": h1,
            "text": "\n".join(lines)}

if __name__ == "__main__":
    urls = sitemap_urls()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(urls)
    ok = fail = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for u in urls[:limit]:
            try:
                r = requests.get(u, headers=UA, timeout=30)
                if r.status_code != 200:
                    fail += 1; print("skip", r.status_code, u); continue
                rec = clean(r.text, u)
                kind = u.rstrip("/").split("/")
                rec["kind"] = kind[3] if len(kind) > 3 else "home"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); ok += 1
            except Exception as e:
                fail += 1; print("err", u, e)
            time.sleep(0.4)
    print("ok", ok, "fail", fail)
