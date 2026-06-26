"""Pull a credible frequency-ranked word list and save it locally.

Primary source: **wordfreq** (Robyn Speer) — aggregates Google Books, Wikipedia,
SUBTLEX (Brysbaert & New), OpenSubtitles, Reddit, etc. Apache-2.0 code,
CC-BY-SA data, 40+ languages (serves the multilingual goal).

Scientist-curated single-corpus alternative: SUBTLEX-US (60,384 words, Brysbaert
& New 2009) — https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus

Writes:
  data/words_<lang>.txt       one word per line, frequency-ranked (the input list)
  data/words_<lang>.meta.json provenance (source, count, filters, timestamp)

The words file is the IMMUTABLE input. Processed-state is NOT tracked here — it
lives in the SQLite manifest (wordbank.db) so resume/attempts/errors survive
crashes and parallel workers. See the note in WORDBANK_PLAN.md.

Usage:
  python scripts/fetch_wordlist.py --lang en --n 50000
"""
import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.wordlist import clean_words


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch a frequency word list via wordfreq")
    p.add_argument("--lang", default="en", help="language code (en, es, fr, de, ...)")
    p.add_argument("--n", type=int, default=50_000, help="how many top words to request")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--allow-nonalpha", action="store_true",
                   help="keep numbers/contractions (default: alpha-only)")
    args = p.parse_args()

    try:
        import importlib.metadata
        import wordfreq
    except ImportError:
        sys.exit("wordfreq not installed. Run: pip install wordfreq")
    try:
        wf_version = importlib.metadata.version("wordfreq")
    except importlib.metadata.PackageNotFoundError:
        wf_version = "unknown"

    raw = wordfreq.top_n_list(args.lang, args.n)
    words = clean_words(
        raw, min_len=args.min_len, max_len=args.max_len,
        alpha_only=not args.allow_nonalpha,
    )

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    words_path = out_dir / f"words_{args.lang}.txt"
    meta_path = out_dir / f"words_{args.lang}.meta.json"

    words_path.write_text("\n".join(words) + "\n", encoding="utf-8")
    meta = {
        "source": "wordfreq",
        "wordfreq_version": wf_version,
        "lang": args.lang,
        "requested": args.n,
        "returned_raw": len(raw),
        "written": len(words),
        "filters": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "alpha_only": not args.allow_nonalpha,
        },
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(words)} words -> {words_path}")
    print(f"meta -> {meta_path}")
    if len(raw) < args.n:
        print(f"note: wordfreq returned {len(raw)} words (< requested {args.n}); "
              "that is the full available list for this language.")


if __name__ == "__main__":
    main()
