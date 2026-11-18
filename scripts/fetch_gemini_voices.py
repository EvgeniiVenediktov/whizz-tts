"""Refresh the curated Gemini voices (wordbank/gemini_voices.json) from Gemini's
voice library, or list candidates when picking a new one.

The repo keeps one female + one male voice per language, not the library
(2,140 voices in 2026-11). Which voice fills each slot is edited by hand in the
JSON (its `id`); this script rewrites every pick's catalog metadata (gender,
display name, accent, pitch, persona, description) and the fetch date, and
refuses to write if a pick is gone or a language no longer has exactly one
female and one male voice.

  GEMINI_API_KEY=… python scripts/fetch_gemini_voices.py           # refresh, live
  python scripts/fetch_gemini_voices.py --from-file voices.json     # refresh, offline
  python scripts/fetch_gemini_voices.py --candidates en-GB --gender male

Live: GET /v1beta/voices (key in x-goog-api-key), paged on `next_page_token`.
--from-file takes a saved {"voices": [...]} holding every page. Lists voices
only: nothing is synthesized or billed.
"""
import argparse
import datetime
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.gemini_tts import GEMINI_API_BASE, GEMINI_API_VERSION  # noqa: E402
from wordbank.languages import VOICE_LANGUAGES_FILE  # noqa: E402

# The catalog fields kept per pick, in file order (`id` is the pick itself).
FIELDS = ("gender", "display_name", "language_code", "accent", "pitch", "persona",
          "description")


def fetch_voices(key: str, opener=urllib.request.urlopen) -> list:
    """Every voice of the library, following next_page_token across pages."""
    voices, token = [], None
    while True:
        query = {"page_size": 1000, **({"page_token": token} if token else {})}
        req = urllib.request.Request(
            f"{GEMINI_API_BASE}/{GEMINI_API_VERSION}/voices?{urllib.parse.urlencode(query)}",
            headers={"x-goog-api-key": key})
        with opener(req, timeout=60) as resp:
            page = json.loads(resp.read())
        voices += page.get("voices") or []
        token = page.get("next_page_token") or page.get("nextPageToken")
        if not token:
            return voices


def refresh(curated: dict, voices: list, fetched: str) -> dict:
    """`curated` with each pick's metadata replaced from `voices` (the library).
    Raises ValueError naming every missing pick or unbalanced language."""
    by_id = {v["id"]: v for v in voices}
    problems, languages = [], []
    for lang in curated["languages"]:
        picks = []
        for pick in lang["voices"]:
            v = by_id.get(pick["id"])
            if v is None:
                problems.append(f"{lang['code']}: {pick['id']} is not in the library")
                continue
            picks.append({"id": v["id"], **{k: v.get(k) for k in FIELDS}})
        if sorted(p["gender"] for p in picks) != ["female", "male"]:
            problems.append(f"{lang['code']}: needs one female and one male voice, got "
                            f"{[(p['id'], p['gender']) for p in picks]}")
        picks.sort(key=lambda p: p["gender"])          # female first
        languages.append({"code": lang["code"], "name": lang["name"], "voices": picks})
    if problems:
        raise ValueError("; ".join(problems))
    return {"source": f"GET {GEMINI_API_BASE}/{GEMINI_API_VERSION}/voices",
            "fetched": fetched, "languages": languages}


def candidates(voices: list, locale: str, gender: str = None) -> list:
    """One line per library voice of `locale` (case-insensitive), optionally of
    one gender, sorted by id."""
    rows = [v for v in voices if (v.get("language_code") or "").lower() == locale.lower()
            and (gender is None or v.get("gender") == gender)]
    return [f"{v['id']:24} {v.get('gender', ''):7} {v.get('pitch', ''):6} "
            f"{v.get('accent', '')} | {v.get('display_name', '')} | {v.get('description', '')}"
            for v in sorted(rows, key=lambda v: v["id"])]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--from-file", dest="from_file", default=None,
                   help='saved library {"voices": [...]} instead of the live API')
    p.add_argument("--candidates", metavar="LOCALE", default=None,
                   help="list the library's voices for a locale (e.g. en-GB), write nothing")
    p.add_argument("--gender", choices=("female", "male", "neutral"), default=None)
    p.add_argument("--out", default=str(VOICE_LANGUAGES_FILE),
                   help="curated file to refresh in place (default: %(default)s)")
    p.add_argument("--fetched", default=None,
                   help="fetch date to record (default: today, or --from-file's mtime)")
    args = p.parse_args(argv)

    if args.from_file:
        voices = json.loads(pathlib.Path(args.from_file).read_text(encoding="utf-8"))["voices"]
        mtime = os.path.getmtime(args.from_file)
        fetched = args.fetched or datetime.date.fromtimestamp(mtime).isoformat()
    else:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            sys.exit("no Gemini API key: set GEMINI_API_KEY (or GOOGLE_API_KEY), or pass --from-file")
        voices = fetch_voices(key)
        fetched = args.fetched or datetime.date.today().isoformat()

    if args.candidates:
        print("\n".join(candidates(voices, args.candidates, args.gender)))
        return
    out = pathlib.Path(args.out)
    try:
        data = refresh(json.loads(out.read_text(encoding="utf-8")), voices, fetched)
    except ValueError as exc:
        sys.exit(f"not written: {exc}")
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}: {sum(len(l['voices']) for l in data['languages'])} voices in "
          f"{len(data['languages'])} languages, from {len(voices)} in the library ({fetched})")


if __name__ == "__main__":
    main()
