#!/usr/bin/env python3
"""Claims linter for Research-docs/v7 — the runnable anti-hallucination gate.

Rules enforced on every .md file in Research-docs/v7/ (excluding tools/):
  R1. Every "claim line" (a line, outside code fences, containing a number-with-unit,
      a currency amount, or a license keyword) must carry a status tag on the same line.
      Status tags: [verified | <url> | as-of YYYY-MM-DD | "quote"]
                   [estimate | <derivation>]
                   [hypothesis | <how to test>]
                   [not-found-as-of YYYY-MM-DD | queries: ...]
                   [inherited-unverified]  (legacy v5/v6 claims quoted for audit)
                   [rumor | <where it circulates>]
                   [internal | <repo-relative path>]  (facts about this repo itself)
  R2. Every [verified ...] tag must contain an http(s) URL and an as-of date.
  R3. Every http(s) URL in v7 docs must be alive (HTTP < 400; 403/405/429 = WARN, not FAIL;
      DNS failure / timeout / 404 / 410 = FAIL). Checked once per unique URL.
Exit code: 0 = all rules pass, 1 = violations found.
"""
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console guard

V7_DIR = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(
    r"\[(verified|estimate|hypothesis|not-found-as-of[^\]|]*|inherited-unverified|rumor|internal|registry)\b[^\]]*\]",
    re.IGNORECASE,
)
VERIFIED_RE = re.compile(r"\[verified\b[^\]]*\]", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\)\]\|>\"',]+")
ASOF_RE = re.compile(r"as-of\s+20\d\d-\d\d-\d\d", re.IGNORECASE)
NUMBER_CLAIM_RE = re.compile(
    r"(?<![\w/.-])\d+(?:[.,]\d+)?\s?(?:%|GB|TB|MB|KB|PB|FPS|fps|dB|ms\b|Hz|MP\b|"
    r"EUR|USD|PLN|€|\$|hours?\b|weeks?\b|months?\b|cameras?\b|subjects?\b|params?\b|"
    r"gaussians?\b|splats?\b|frames?\b|views?\b|min\b|sec\b|s\b)",
)
LICENSE_CLAIM_RE = re.compile(
    r"\b(Apache[- ]2\.0|MIT license|GPL[- ]?3|GPLv3|CC[- ]BY|non-?commercial|research[- ]only|"
    r"proprietary license|commercial license|commercially clean)\b",
    re.IGNORECASE,
)
SKIP_MARK = "lint-skip"


def iter_md_files():
    for p in sorted(V7_DIR.glob("*.md")):
        yield p


def effective_lines(text):
    """Lines outside code fences and not marked lint-skip."""
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or SKIP_MARK in line:
            continue
        yield i, line


def claim_lines(text):
    for i, line in effective_lines(text):
        if line.strip().startswith("#"):  # headings are structure, not claims
            continue
        if NUMBER_CLAIM_RE.search(line) or LICENSE_CLAIM_RE.search(line):
            yield i, line


def check_url(url, cache={}):
    if url in cache:
        return cache[url]
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0 (claims-linter)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = -1
    if status == -1 or status in (404, 410):
        verdict = "FAIL"
    elif status >= 400:
        verdict = "WARN"  # 403/405/429: bot-blocked but resolvable
    else:
        verdict = "OK"
    cache[url] = (verdict, status)
    return verdict, status


def main():
    check_urls = "--no-url-check" not in sys.argv
    failures, warns = [], []
    all_urls = set()
    for path in iter_md_files():
        text = path.read_text(encoding="utf-8")
        all_urls.update(URL_RE.findall(text))
        for lineno, line in claim_lines(text):
            if not TAG_RE.search(line):
                failures.append(f"{path.name}:{lineno}: untagged claim: {line.strip()[:110]}")
        for lineno, line in effective_lines(text):
            for m in VERIFIED_RE.finditer(line):
                tag = m.group(0)
                if not URL_RE.search(tag):
                    failures.append(f"{path.name}:{lineno}: [verified] tag without URL: {tag[:90]}")
                if not ASOF_RE.search(tag):
                    failures.append(f"{path.name}:{lineno}: [verified] tag without as-of date: {tag[:90]}")
    if check_urls:
        for url in sorted(all_urls):
            verdict, status = check_url(url.rstrip(".;:"))
            if verdict == "FAIL":
                failures.append(f"dead URL ({status}): {url}")
            elif verdict == "WARN":
                warns.append(f"bot-blocked URL ({status}), verify manually: {url}")
    for w in warns:
        print(f"WARN  {w}")
    for f in failures:
        print(f"FAIL  {f}")
    print(f"\nchecked files: {len(list(iter_md_files()))}, unique URLs: {len(all_urls)}, "
          f"failures: {len(failures)}, warnings: {len(warns)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
