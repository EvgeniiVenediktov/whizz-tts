"""Word-bank CLI:  python -m wordbank <command> ...

Commands: batch / batch-submit / batch-collect / batch-run (Gemini async) ·
gemini-run (Gemini sync, rate-limited) · regenerate (back up + reset words) ·
describe / verify (deployment library description) · status · export.

  python -m wordbank batch-run --words words.txt --voice Kore
  python -m wordbank --lang en-gb gemini-run --model gemini-3.8-flash-tts \
      --voice Kore --style "plain and neutral"
  python -m wordbank status
"""
import argparse
import csv
import os
import pathlib
import sys
from typing import List, Optional, Tuple

from .config import Config
from .gemini_tts import DEFAULT_MODEL as GEMINI_TTS_DEFAULT_MODEL   # stdlib-only import
from .languages import normalize_lang
from .manifest import Manifest


def _load_words(path: str) -> List[str]:
    seen, words = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#") or w in seen:
                continue
            seen.add(w)
            words.append(w)
    return words


def _ingest(cfg: Config, words_path: str) -> Manifest:
    """Load a word file into the manifest (deduped) and report what's new."""
    words = _load_words(words_path)
    m = Manifest(cfg.db)
    added = m.add_words(words, cfg.lang)
    print(f"{len(words)} words in file; {added} newly added to manifest")
    return m


def _add_direction_args(p) -> None:
    p.add_argument("--instruction", default=None,
                   help="delivery direction. 3.1/2.5 models: prepended to each word "
                        "(default: gemini_batch.VOICE_INSTRUCTION). 3.8 models: sent as "
                        "the style, never inline (they read inline text aloud), and no "
                        "default. --instruction= sends the bare word")
    p.add_argument("--style", default=None,
                   help="delivery direction sent as speech metadata, the word kept "
                        "verbatim; 3.8 models only (see gemini_tts.TTS_MODELS)")


def _add_gemini_args(p, *, words=True, instruction=True, chunk=True, poll=True):
    """Common args for the Gemini batch subcommands (which differ only slightly)."""
    if words:
        p.add_argument("--words", required=True, help="text file, one word per line")
    p.add_argument("--voice", default="Kore", help="Gemini voice name")
    p.add_argument("--model", default="gemini-3.1-flash-tts-preview")
    if instruction:
        _add_direction_args(p)
    if chunk:
        p.add_argument("--chunk", type=int, default=2000, help="words per batch job")
    if poll:
        p.add_argument("--poll-interval", type=int, default=60, dest="poll_interval")


def _direction_args(args) -> Tuple[str, Optional[str]]:
    """(instruction, style) for a Gemini generation command; exits on a
    combination that would be rejected or spoken. An unset --instruction
    defaults to VOICE_INSTRUCTION only where it stays unspoken (inline-capable
    models): a 3.8 model needs an explicit --style or --instruction."""
    from .gemini_batch import VOICE_INSTRUCTION
    from .gemini_tts import model_caps, style_models

    caps = model_caps(args.model)
    style = (args.style or "").strip() or None
    if style and not caps.style:
        sys.exit(f"--style needs a style-capable model ({', '.join(style_models())}); "
                 f"{args.model} takes inline directions: use --instruction")
    if args.instruction is not None:
        return args.instruction, style
    if style:
        return "", style
    if caps.inline:
        return VOICE_INSTRUCTION, None
    sys.exit(f"{args.model} reads inline instructions aloud, so the default --instruction "
             "is not sent to it: pass --style \"...\" for a delivery direction (speech "
             "metadata), or --instruction= for the bare word")


def _make_batch(args, *, submit: bool = True):
    """The GeminiBatch for a command; `submit=False` (collect only) needs no
    direction, since no request is built."""
    from .gemini_batch import GeminiBatch
    if not submit:
        return GeminiBatch(model=args.model, voice=args.voice, instruction="")
    instruction, style = _direction_args(args)
    return GeminiBatch(model=args.model, voice=args.voice, instruction=instruction,
                       style=style)


def cmd_batch_submit(args, cfg: Config) -> None:
    """Submit pending words as async job(s) and exit — safe to power off after."""
    from .gemini_batch import submit_pending

    batch = _make_batch(args)              # refuses a bad direction before ingesting
    m = _ingest(cfg, args.words)
    jobs = submit_pending(m, batch, lang=cfg.lang, chunk=args.chunk)
    print(f"\nsubmitted {len(jobs)} batch job(s): {jobs}")
    print("job ids saved in the manifest. Collect later (jobs expire after 48h) with:")
    print(f"  python -m wordbank --db {cfg.db} --out {cfg.out_dir} "
          f"batch-collect --voice {args.voice}")
    m.close()


def cmd_batch_collect(args, cfg: Config) -> None:
    """Poll in-flight job(s) and write finished clips. Run after a reboot."""
    from .gemini_batch import collect_active

    m = Manifest(cfg.db)
    counts = collect_active(
        m, _make_batch(args, submit=False), cfg.out_dir, lang=cfg.lang, voice=args.voice,
        poll_interval=args.poll_interval,
    )
    print(f"totals: {counts}")
    m.close()


def cmd_batch(args, cfg: Config) -> None:
    """Blocking convenience: submit then poll+collect in one process."""
    from .gemini_batch import run_batch

    batch = _make_batch(args)
    m = _ingest(cfg, args.words)
    counts = run_batch(
        m, batch, cfg.out_dir, lang=cfg.lang, voice=args.voice,
        chunk=args.chunk, poll_interval=args.poll_interval,
    )
    print(f"totals: {counts}")
    m.close()


def cmd_batch_run(args, cfg: Config) -> None:
    """Quota-aware waves: submit→collect→repeat until done (keep the laptop on)."""
    from .gemini_batch import run_in_waves

    batch = _make_batch(args)
    m = _ingest(cfg, args.words)
    counts = run_in_waves(
        m, batch, cfg.out_dir, lang=cfg.lang, voice=args.voice,
        chunk=args.chunk, poll_interval=args.poll_interval,
    )
    print(f"totals: {counts}")
    m.close()


def cmd_gemini_run(args, cfg: Config) -> None:
    """Sync, rate-limited Gemini generation of pending words (resumable)."""
    from .gemini_run import run_sync

    instruction, style = _direction_args(args)
    m = _ingest(cfg, args.words) if args.words else Manifest(cfg.db)
    try:
        result = run_sync(m, cfg.out_dir, lang=cfg.lang, voice=args.voice,
                          model=args.model, instruction=instruction, style=style,
                          rpm=args.rpm, max_attempts=args.max_attempts, limit=args.limit)
    except ValueError as exc:               # an unusable --voice
        sys.exit(str(exc))
    finally:
        totals = m.counts()
        m.close()
    print(f"this run: {result}")
    print(f"totals: {totals}")
    if result["stopped"]:
        sys.exit(1)


def cmd_regenerate(args, cfg: Config) -> None:
    """Move the current take of each listed word to <db dir>/takes/<UTC>/ and
    reset its row to pending, for gemini-run / batch-run to generate again."""
    import datetime
    import json
    import shutil
    from .paths import safe_filename

    words = _load_words(args.words_file)
    if not os.path.isfile(cfg.db):
        sys.exit(f"no word bank at {cfg.db} (pass --db)")
    m = Manifest(cfg.db)
    try:
        rows, unknown = [], []
        for w in words:
            row = m.get_entry(w, cfg.lang)
            if row is None:
                unknown.append(w)
            else:
                rows.append(row)
        for w in unknown:
            print(f"unknown (not in the {cfg.lang} word bank, not created): {w}")
        if not rows:
            sys.exit(f"none of the {len(words)} word(s) in {args.words_file} is in the "
                     f"{cfg.lang} word bank")

        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        take_dir = pathlib.Path(cfg.db).resolve().parent / "takes" / stamp
        roots = [("wav", cfg.out_dir, ".wav")]
        if args.ogg_root:
            roots.append(("ogg", args.ogg_root, ".ogg"))
        listing = []
        for row in rows:
            entry = {k: row[k] for k in ("id", "word", "status", "voice", "model",
                                         "style", "file_path")}
            for kind, root, ext in roots:
                entry[kind] = None
                if not row["file_path"]:
                    continue
                src = pathlib.Path(root) / (os.path.splitext(row["file_path"])[0] + ext)
                if src.is_file():
                    # moved, not copied: encode_wavs.py skips existing OGGs, so
                    # a stale copy left in place would never be re-encoded
                    dest = take_dir / kind / f"{row['id']}_{safe_filename(row['word'])}{ext}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest))
                    entry[kind] = str(dest.relative_to(take_dir))
            listing.append(entry)
        m.reset([r["id"] for r in rows])
        take_dir.mkdir(parents=True, exist_ok=True)
        (take_dir / "words.json").write_text(json.dumps(listing, indent=1) + "\n",
                                             encoding="utf-8")
    finally:
        m.close()

    moved = {kind: sum(1 for e in listing if e.get(kind)) for kind, _, _ in roots}
    print(f"reset {len(rows)} word(s) to pending; {len(unknown)} unknown")
    print(f"previous take: {take_dir} ({', '.join(f'{n} {k}' for k, n in moved.items())} "
          "moved, listing in words.json)")
    voices = sorted({r["voice"] for r in rows if r["voice"]})
    print("generate them again, e.g.:")
    print(f"  python -m wordbank --db {cfg.db} --out {cfg.out_dir} --lang {cfg.lang} "
          f"gemini-run --voice {voices[0] if len(voices) == 1 else '<voice>'} "
          "--model <model> [--style \"...\"]")
    if args.ogg_root:
        print(f"  python scripts/encode_wavs.py {cfg.out_dir} {args.ogg_root}")


def cmd_describe(args, cfg: Config) -> None:
    """Emit a library description (languages/counts/folders) as JSON."""
    import json
    from .library import describe

    desc = describe(cfg.db, args.ogg_root)
    text = json.dumps(desc, indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}: {desc['words']} words, {desc['phrases']} phrases, "
              f"{desc['ogg_files']} ogg")
    else:
        print(text)


def cmd_verify(args, cfg: Config) -> None:
    """Verify a mounted volume against a library description; exit 1 if invalid."""
    from .library import verify

    problems = verify(args.desc, args.root, db=args.db_override,
                      ogg_root=args.ogg_root)
    if problems:
        print("library INVALID:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    where = args.root if args.root is not None else "default layout"
    print(f"library OK: {where} matches {args.desc}")


def cmd_status(args, cfg: Config) -> None:
    words = Manifest(cfg.db)
    phrases = Manifest(cfg.db, table="phrases", text_col="text")
    print(f"words:   {words.counts()}")
    print(f"phrases: {phrases.counts()}")
    words.close()
    phrases.close()


def cmd_export(args, cfg: Config) -> None:
    m = Manifest(cfg.db)
    rows = m.iter_rows()
    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    writer = csv.writer(out)
    writer.writerow(["id", "word", "lang", "status", "attempts", "file_path",
                     "transcript", "billed_chars", "error", "model", "style"])
    for r in rows:
        writer.writerow([r["id"], r["word"], r["lang"], r["status"], r["attempts"],
                         r["file_path"], r["transcript"], r["billed_chars"], r["error"],
                         r["model"], r["style"]])
    if args.output:
        out.close()
        print(f"wrote {args.output}")
    m.close()


def main(argv=None) -> None:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(prog="wordbank")
    p.add_argument("--db", default=cfg.db)
    p.add_argument("--out", default=cfg.out_dir, dest="out_dir")
    p.add_argument("--lang", default=cfg.lang, type=normalize_lang)  # a path segment of every clip
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("batch", help="generate via Gemini Batch API (async, ~half cost)")
    _add_gemini_args(g)
    g.set_defaults(func=cmd_batch)

    bs = sub.add_parser("batch-submit",
                        help="submit async job(s) and exit — safe to power off after")
    _add_gemini_args(bs, poll=False)
    bs.set_defaults(func=cmd_batch_submit)

    bc = sub.add_parser("batch-collect",
                        help="poll in-flight job(s) and write clips (run in the morning)")
    _add_gemini_args(bc, words=False, instruction=False, chunk=False)
    bc.set_defaults(func=cmd_batch_collect)

    br = sub.add_parser("batch-run",
                        help="quota-aware waves: submit+collect+repeat until done (laptop stays on)")
    _add_gemini_args(br)
    br.set_defaults(func=cmd_batch_run)

    gr = sub.add_parser("gemini-run",
                        help="generate pending words with Gemini one request at a time "
                             "(rate-limited, resumable; supports --style)")
    gr.add_argument("--words", default=None,
                    help="text file, one word per line, added first (default: just "
                         "generate what the manifest has pending, e.g. after regenerate)")
    gr.add_argument("--voice", default="Kore", help="Gemini voice name")
    gr.add_argument("--model", default=GEMINI_TTS_DEFAULT_MODEL,
                    help=f"Gemini TTS model (default {GEMINI_TTS_DEFAULT_MODEL})")
    _add_direction_args(gr)
    gr.add_argument("--rpm", type=float, default=10,
                    help="requests per minute (default 10 = Tier 1), retries included")
    gr.add_argument("--max-attempts", type=int, default=3, dest="max_attempts",
                    help="attempts per word across runs (default 3)")
    gr.add_argument("--limit", type=int, default=None, help="cap words this run")
    gr.set_defaults(func=cmd_gemini_run)

    rg = sub.add_parser("regenerate",
                        help="move the current take of listed words to <db dir>/takes/<UTC>/ "
                             "and reset them to pending")
    rg.add_argument("--words-file", required=True, dest="words_file",
                    help="text file, one word per line (words not in the bank are reported)")
    rg.add_argument("--ogg-root", default=None, dest="ogg_root",
                    help="OGG tree to take the encoded copies from too, e.g. output/words_ogg")
    rg.set_defaults(func=cmd_regenerate)

    de = sub.add_parser("describe",
                        help="write a library description (langs/counts) as JSON")
    de.add_argument("--ogg-root", default="output/words_ogg", dest="ogg_root")
    de.add_argument("--output", default=None, help="path to write (default: stdout)")
    de.set_defaults(func=cmd_describe)

    ve = sub.add_parser("verify",
                        help="verify a volume against a library description (exit 1 if invalid)")
    ve.add_argument("--desc", required=True, help="library description JSON")
    ve.add_argument("--root", default=None,
                    help="flat deploy layout data dir (db + words_ogg/ under here, as "
                         "docker-entrypoint.sh lays it out). Omit to check the "
                         "dev/server default layout ($BANK_DB, $OGG_ROOT).")
    ve.add_argument("--db", default=None, dest="db_override",
                    help="override the db path (wins over --root and $BANK_DB)")
    ve.add_argument("--ogg-root", default=None, dest="ogg_root",
                    help="override the OGG tree (wins over --root and $OGG_ROOT)")
    ve.set_defaults(func=cmd_verify)

    s = sub.add_parser("status", help="show counts by status")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("export", help="export manifest to CSV")
    e.add_argument("--output", default=None, help="CSV path (default: stdout)")
    e.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    cfg = Config(db=args.db, out_dir=args.out_dir, lang=args.lang)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
