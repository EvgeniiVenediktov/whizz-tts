"""Word-bank CLI:  python -m wordbank <command> ...

Commands: generate (local OmniVoice) · batch / batch-submit / batch-collect /
batch-run (Gemini async) · status · export.

  python -m wordbank --lang en-gb generate --words phrases.txt \
      --ref-audio refs/en-gb.wav --ref-text "…" --voice en-gb
  python -m wordbank batch-run --words words.txt --voice Kore
  python -m wordbank status
"""
import argparse
import csv
import pathlib
import sys
from typing import List

from .config import Config
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


def _add_gemini_args(p, *, words=True, instruction=True, chunk=True, poll=True):
    """Common args for the Gemini batch subcommands (which differ only slightly)."""
    if words:
        p.add_argument("--words", required=True, help="text file, one word per line")
    p.add_argument("--voice", default="Kore", help="Gemini prebuilt voice name")
    p.add_argument("--model", default="gemini-3.1-flash-tts-preview")
    if instruction:
        p.add_argument("--instruction", default=None,
                       help="style direction prepended to each word ('' = bare word)")
    if chunk:
        p.add_argument("--chunk", type=int, default=2000, help="words per batch job")
    if poll:
        p.add_argument("--poll-interval", type=int, default=60, dest="poll_interval")


def cmd_generate(args, cfg: Config) -> None:
    """Generate phrases locally with OmniVoice; accent via a clone reference."""
    from .backends.local import LocalOmniVoiceBackend
    from .pipeline import WordBankPipeline

    ref_text = args.ref_text
    if ref_text is None and args.ref_text_file:
        ref_text = pathlib.Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    backend = LocalOmniVoiceBackend(ref_audio=args.ref_audio, ref_text=ref_text,
                                    voice=args.voice)
    verifier = None
    if args.verify:
        from .verify import Verifier
        verifier = Verifier()

    m = _ingest(cfg, args.words)
    pipe = WordBankPipeline(m, backend, verifier, cfg.out_dir,
                            lang=cfg.lang, voice=args.voice)
    result = pipe.run(limit=args.limit)   # sequential — the model isn't thread-safe
    print(f"this run: {result}")
    print(f"totals:   {m.counts()}")
    m.close()


def _make_batch(args):
    from .gemini_batch import VOICE_INSTRUCTION, GeminiBatch
    instruction = VOICE_INSTRUCTION if getattr(args, "instruction", None) is None \
        else args.instruction
    return GeminiBatch(model=args.model, voice=args.voice, instruction=instruction)


def cmd_batch_submit(args, cfg: Config) -> None:
    """Submit pending words as async job(s) and exit — safe to power off after."""
    from .gemini_batch import submit_pending

    m = _ingest(cfg, args.words)
    jobs = submit_pending(m, _make_batch(args), lang=cfg.lang, chunk=args.chunk)
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
        m, _make_batch(args), cfg.out_dir, lang=cfg.lang, voice=args.voice,
        poll_interval=args.poll_interval,
    )
    print(f"totals: {counts}")
    m.close()


def cmd_batch(args, cfg: Config) -> None:
    """Blocking convenience: submit then poll+collect in one process."""
    from .gemini_batch import run_batch

    m = _ingest(cfg, args.words)
    counts = run_batch(
        m, _make_batch(args), cfg.out_dir, lang=cfg.lang, voice=args.voice,
        chunk=args.chunk, poll_interval=args.poll_interval,
    )
    print(f"totals: {counts}")
    m.close()


def cmd_batch_run(args, cfg: Config) -> None:
    """Quota-aware waves: submit→collect→repeat until done (keep the laptop on)."""
    from .gemini_batch import run_in_waves

    m = _ingest(cfg, args.words)
    counts = run_in_waves(
        m, _make_batch(args), cfg.out_dir, lang=cfg.lang, voice=args.voice,
        chunk=args.chunk, poll_interval=args.poll_interval,
    )
    print(f"totals: {counts}")
    m.close()


def cmd_status(args, cfg: Config) -> None:
    m = Manifest(cfg.db)
    print(m.counts())
    m.close()


def cmd_export(args, cfg: Config) -> None:
    m = Manifest(cfg.db)
    rows = m.iter_rows()
    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    writer = csv.writer(out)
    writer.writerow(["id", "word", "lang", "status", "attempts", "file_path",
                     "transcript", "billed_chars", "error"])
    for r in rows:
        writer.writerow([r["id"], r["word"], r["lang"], r["status"], r["attempts"],
                         r["file_path"], r["transcript"], r["billed_chars"], r["error"]])
    if args.output:
        out.close()
        print(f"wrote {args.output}")
    m.close()


def main(argv=None) -> None:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(prog="wordbank")
    p.add_argument("--db", default=cfg.db)
    p.add_argument("--out", default=cfg.out_dir, dest="out_dir")
    p.add_argument("--lang", default=cfg.lang)
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate",
                         help="generate phrases locally with OmniVoice (accent via clone reference)")
    gen.add_argument("--words", required=True, help="text file, one phrase/word per line")
    gen.add_argument("--voice", default="local", help="label for the output dir, e.g. en-gb")
    gen.add_argument("--ref-audio", default=None, dest="ref_audio",
                     help="voice-clone reference clip (the accent), e.g. a Gemini phrase")
    gen.add_argument("--ref-text", default=None, dest="ref_text",
                     help="transcript of --ref-audio")
    gen.add_argument("--ref-text-file", default=None, dest="ref_text_file",
                     help="file holding the transcript of --ref-audio")
    gen.add_argument("--limit", type=int, default=None, help="cap phrases this run")
    gen.add_argument("--verify", action="store_true",
                     help="STT-verify each clip with local Whisper (slow)")
    gen.set_defaults(func=cmd_generate)

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
