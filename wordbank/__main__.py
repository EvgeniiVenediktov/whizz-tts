"""Word-bank CLI:  python -m wordbank <command> ...

Commands: batch-submit / batch-collect / batch-run (Gemini async) ·
gemini-run (Gemini sync, rate-limited) · regenerate (back up + reset words) ·
qc (check clips, re-take flagged ones) · describe / verify (deployment library
description) · status · export.

  python -m wordbank batch-run --words words.txt --voice Kore
  python -m wordbank --lang en-gb gemini-run --model gemini-3.8-flash-tts \
      --voice Kore --style "plain and neutral"
  python -m wordbank --lang en-gb qc --csv review.csv
  python -m wordbank status
"""
import argparse
import csv
import json
import os
import pathlib
import sys
from typing import List, Optional, Tuple

from .config import Config
from .gemini_tts import DEFAULT_MODEL as GEMINI_TTS_DEFAULT_MODEL, voice_label   # stdlib-only
from .languages import normalize_lang
from .manifest import RETAKE, Manifest
from .qc import DEFAULT_RETRIES as QC_DEFAULT_RETRIES


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


def _ingest(cfg: Config, words_path: str, voice: str) -> Manifest:
    """Load a word file into the manifest for one voice (deduped per voice:
    a word already banked in another voice gets a row of its own) and report
    what's new."""
    words = _load_words(words_path)
    m = Manifest(cfg.db)
    added = m.add_words(words, cfg.lang, voice=voice)
    print(f"{len(words)} words in file; {added} newly added to manifest for voice {voice}")
    return m


def _add_qc_arg(p) -> None:
    p.add_argument("--qc-retries", type=int, default=QC_DEFAULT_RETRIES, dest="qc_retries",
                   help="extra takes for a clip QC flags (wordbank/qc.py: ends on sound, "
                        f"or an implausible length); 0 = only flag (default {QC_DEFAULT_RETRIES})")


def _add_direction_args(p) -> None:
    p.add_argument("--instruction", default=None,
                   help="delivery direction. 3.1/2.5 models: prepended to each word "
                        "(default: gemini_batch.VOICE_INSTRUCTION). 3.8 models: sent as "
                        "the style, never inline (they read inline text aloud), and no "
                        "default. --instruction= sends the bare word")
    p.add_argument("--style", default=None,
                   help="delivery direction sent as speech metadata, the word kept "
                        "verbatim; 3.8 models only (see gemini_tts.TTS_MODELS)")


def _add_rpm_arg(p) -> None:
    p.add_argument("--rpm", type=float, default=10,
                   help="requests per minute (default 10 = Tier 1), retries included")


_VOICE_HELP = ("Gemini voice: a classic name (Kore), or a library id like en-gb-tutor-11 "
               "(3.8 models only), filed as gemini-<voice>")


def _add_gemini_args(p, *, submit=True, poll=True):
    """Args of the Gemini batch subcommands: submitting ones take the words,
    voice, model and direction; collecting ones (poll) --qc-retries."""
    if submit:
        p.add_argument("--words", required=True, help="text file, one word per line")
        p.add_argument("--voice", default="Kore", type=voice_label, help=_VOICE_HELP)
        p.add_argument("--model", default="gemini-3.1-flash-tts-preview")
        _add_direction_args(p)
        p.add_argument("--chunk", type=int, default=2000, help="words per batch job")
    if poll:
        p.add_argument("--poll-interval", type=int, default=60, dest="poll_interval")
        _add_qc_arg(p)


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
    if not caps.style:
        return VOICE_INSTRUCTION, None
    sys.exit(f"{args.model} reads inline instructions aloud, so the default --instruction "
             "is not sent to it: pass --style \"...\" for a delivery direction (speech "
             "metadata), or --instruction= for the bare word")


def _check_voice(voice: str, model: str, flag: str = "--voice") -> None:
    """Exit before anything is ingested or sent when `voice` isn't one path
    segment, or is a locale voice `model` isn't verified with (check_voice)."""
    from .gemini_tts import check_voice
    from .paths import check_segment
    try:
        check_segment(voice)
        check_voice(voice, model)
    except ValueError as exc:               # GeminiVoiceUnsupportedError included
        sys.exit(f"{flag}: {exc}")


def _require_db(cfg: Config) -> None:
    if not os.path.isfile(cfg.db):
        sys.exit(f"no word bank at {cfg.db} (pass --db)")


def _print_totals(m: Manifest) -> None:
    """Status counts, QC flags, and what to do about rows queued for a re-take."""
    counts, qcc = m.counts(), m.qc_counts()
    print(f"totals: {counts}")
    print(f"QC: {qcc['flagged']} clip(s) flagged for review, {qcc['unchecked']} never checked")
    if counts.get(RETAKE):
        print(f"{counts[RETAKE]} row(s) flagged by QC are queued for a re-take (status "
              f"{RETAKE}): run batch-submit or batch-run again with the same --voice, --model "
              "and --style/--instruction, or re-take them now with `qc --retry`")


def _make_batch(args):
    """The GeminiBatch a submitting command sends with (voice and direction checked)."""
    from .gemini_batch import GeminiBatch
    _check_voice(args.voice, args.model)
    instruction, style = _direction_args(args)
    return GeminiBatch(model=args.model, voice=args.voice, instruction=instruction,
                       style=style)


def cmd_batch_submit(args, cfg: Config) -> None:
    """Submit pending words as async job(s) and exit — safe to power off after."""
    from .gemini_batch import submit_pending

    batch = _make_batch(args)              # refuses a bad direction before ingesting
    m = _ingest(cfg, args.words, args.voice)
    jobs = submit_pending(m, batch, lang=cfg.lang, chunk=args.chunk)
    print(f"\nsubmitted {len(jobs)} batch job(s): {jobs}")
    print("job ids saved in the manifest. Collect later (jobs expire after 48h) with:")
    print(f"  python -m wordbank --db {cfg.db} --out {cfg.out_dir} batch-collect")
    m.close()


def cmd_batch_collect(args, cfg: Config) -> None:
    """Poll in-flight job(s) and write finished clips. Run after a reboot."""
    from .gemini_batch import GeminiBatch, collect_active

    m = Manifest(cfg.db)
    collect_active(m, GeminiBatch(instruction=""), cfg.out_dir,
                   poll_interval=args.poll_interval, qc_retries=args.qc_retries)
    _print_totals(m)
    m.close()


def cmd_batch_run(args, cfg: Config) -> None:
    """Quota-aware waves: submit→collect→repeat until done (keep the laptop on)."""
    from .gemini_batch import run_in_waves

    batch = _make_batch(args)
    m = _ingest(cfg, args.words, args.voice)
    run_in_waves(
        m, batch, cfg.out_dir, lang=cfg.lang,
        chunk=args.chunk, poll_interval=args.poll_interval, qc_retries=args.qc_retries,
    )
    _print_totals(m)
    m.close()


def cmd_gemini_run(args, cfg: Config) -> None:
    """Sync, rate-limited Gemini generation of pending words (resumable)."""
    from .gemini_run import run_sync

    _check_voice(args.voice, args.model)
    instruction, style = _direction_args(args)
    m = _ingest(cfg, args.words, args.voice) if args.words else Manifest(cfg.db)
    try:
        result = run_sync(m, cfg.out_dir, lang=cfg.lang, voice=args.voice,
                          model=args.model, instruction=instruction, style=style,
                          rpm=args.rpm, max_attempts=args.max_attempts, limit=args.limit,
                          qc_retries=args.qc_retries)
        print(f"this run: {result}")
        _print_totals(m)
    finally:
        m.close()
    if result["stopped"]:
        sys.exit(1)


def _take_dir(cfg: Config) -> pathlib.Path:
    """A new <db dir>/takes/<UTC>-<random>/: where replaced takes are moved,
    with a words.json."""
    import datetime
    import tempfile
    root = pathlib.Path(cfg.db).resolve().parent / "takes"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=root))


def _take_roots(cfg: Config, ogg_root: Optional[str]) -> list:
    return [("wav", cfg.out_dir, ".wav")] + ([("ogg", ogg_root, ".ogg")] if ogg_root else [])


def _archive_take(cfg: Config, row, take_dir: pathlib.Path, roots, listing: list) -> None:
    """Move a row's current take out of the bank trees into take_dir/{wav,ogg}/
    and add its entry to `listing`, rewriting take_dir/words.json at once (an
    interrupted run still says what moved where). A row queued for a QC
    re-take holds its take outside the bank (qc.retake_path)."""
    import shutil
    from .paths import safe_filename
    from .qc import retake_path

    entry = {k: row[k] for k in ("id", "word", "status", "voice", "model", "style",
                                 "file_path", "review")}
    for kind, root, ext in roots:
        entry[kind] = None
        if kind == "wav" and row["status"] == RETAKE:
            src = retake_path(cfg.db, row["id"])
        elif row["file_path"]:
            src = pathlib.Path(root) / (os.path.splitext(row["file_path"])[0] + ext)
        else:
            continue
        if src.is_file():
            dest = take_dir / kind / f"{row['id']}_{safe_filename(row['word'])}{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            entry[kind] = str(dest.relative_to(take_dir))
    listing.append(entry)
    (take_dir / "words.json").write_text(json.dumps(listing, indent=1) + "\n", encoding="utf-8")


def cmd_regenerate(args, cfg: Config) -> None:
    """Move the current take of each listed word to <db dir>/takes/<UTC>/ and
    reset its row to pending, for gemini-run / batch-run to generate again.
    --voice limits it to that voice's rows; without it, every voice's."""
    words = _load_words(args.words_file)
    _require_db(cfg)
    voice = args.voice
    bank = f"{cfg.lang} word bank" + (f" in voice {voice}" if voice else "")
    m = Manifest(cfg.db)
    try:
        rows, unknown = [], []
        for w in words:
            found = m.entries(w, cfg.lang, voice)
            rows += found
            if not found:
                unknown.append(w)
        for w in unknown:
            print(f"unknown (not in the {bank}, not created): {w}")
        if not rows:
            sys.exit(f"none of the {len(words)} word(s) in {args.words_file} is in the {bank}")

        take_dir, roots, listing = _take_dir(cfg), _take_roots(cfg, args.ogg_root), []
        for row in rows:
            _archive_take(cfg, row, take_dir, roots, listing)
        m.reset([r["id"] for r in rows])
    finally:
        m.close()

    moved = {kind: sum(1 for e in listing if e.get(kind)) for kind, _, _ in roots}
    print(f"reset {len(words) - len(unknown)} word(s) to pending ({len(rows)} row(s), one per "
          f"voice); {len(unknown)} unknown")
    print(f"previous take: {take_dir} ({', '.join(f'{n} {k}' for k, n in moved.items())} "
          "moved, listing in words.json)")
    print("generate them again, e.g.:")
    for v in sorted({r["voice"] or "<voice>" for r in rows}):
        print(f"  python -m wordbank --db {cfg.db} --out {cfg.out_dir} --lang {cfg.lang} "
              f"gemini-run --voice {v} --model <model> [--style \"...\"]")
    if args.ogg_root:
        print(f"  python scripts/encode_wavs.py {cfg.out_dir} {args.ogg_root}")


def cmd_qc(args, cfg: Config) -> None:
    """Check the bank's clips (wordbank/qc.py) and record the result in
    `review` / qc_seconds / qc_end_db. With --retry, re-take each flagged word
    synchronously (gemini-run's rate-limited machinery, the row's own voice)
    until a take passes or --qc-retries run out; the best take is kept, a
    replaced one moved to <db dir>/takes/<UTC>/."""
    import wave
    from . import qc
    from .manifest import DONE

    _require_db(cfg)
    voice = args.voice
    lang = None if args.all_langs else cfg.lang
    scope = f"{lang or 'every language'}, {f'voice {voice}' if voice else 'every voice'}"
    out = pathlib.Path(cfg.out_dir)

    def clip(row) -> pathlib.Path:          # where the row's current take is
        if row["status"] == RETAKE:
            return qc.retake_path(cfg.db, row["id"])
        return out / row["file_path"]

    m = Manifest(cfg.db)
    try:
        in_scope = [r for r in m.iter_rows()
                    if r["status"] in (DONE, RETAKE) and r["file_path"]
                    and (lang is None or r["lang"] == lang) and (voice is None or r["voice"] == voice)]
        if args.accept_file:
            keep = set(_load_words(args.accept_file))
            accept = [r for r in in_scope if r["word"] in keep]
            waiting = [r["word"] for r in accept if r["status"] == RETAKE]
            if waiting:     # its take is held outside the bank: nothing to accept yet
                sys.exit(f"--accept-file: {', '.join(waiting)} wait(s) for a QC re-take "
                         "(status retake); re-take first (qc --retry, or the next batch)")
            m.accept_qc([r["id"] for r in accept])
            print(f"accepted {len(accept)} clip(s) ({scope}); "
                  f"{len(keep - {r['word'] for r in accept})} listed word(s) not in scope")
            in_scope = [dict(r) | {"qc_accepted": 1} if r["word"] in keep else r for r in in_scope]
        rows = [r for r in in_scope if not r["qc_accepted"]]
        checks, missing = {}, []
        for row in rows:
            try:
                checks[row["id"]] = qc.check_wav(clip(row))
            except (OSError, EOFError, wave.Error, ValueError) as exc:
                missing.append((row, exc))
        m.record_qc([(rid, check, None) for rid, check in checks.items()])
        flagged = [r for r in rows if r["id"] in checks and checks[r["id"]].flagged]
        final = {r["id"]: checks[r["id"]] for r in flagged}
        take_dir, replaced = None, 0
        if args.retry and flagged:
            take_dir, replaced = _qc_retakes(args, cfg, m, flagged, checks, final, clip)
        still = [r for r in flagged if final[r["id"]].flagged]
        if args.csv:
            now = {r["id"]: r for r in m.iter_rows()}     # file_path may have moved
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "word", "voice", "seconds", "end_db", "reason", "file_path"])
                for r in still:
                    c, row = final[r["id"]], now[r["id"]]
                    writer.writerow([row["id"], row["word"], row["voice"], f"{c.seconds:.2f}",
                                     f"{c.end_db:.1f}", c.review, row["file_path"]])
    finally:
        m.close()

    for row, exc in missing[:10]:           # a volume without WAV masters misses them all
        print(f"  no readable clip for {row['word']!r} (id {row['id']}): {exc}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more (qc reads the WAV masters under --out)")
    resolved = len(flagged) - len(still)
    print(f"checked {len(checks)} clip(s) ({scope}): {len(flagged)} flagged"
          + (f", {resolved} resolved by a re-take" if args.retry else "")
          + f", {len(still)} still flagged" + (f"; {len(missing)} unreadable" if missing else ""))
    if args.csv:
        print(f"wrote {args.csv} ({len(still)} still flagged)")
    if replaced:
        print(f"replaced {replaced} clip(s); previous takes in {take_dir}")
        print(f"  python scripts/encode_wavs.py {cfg.out_dir} {args.ogg_root or '<ogg root>'}")
    elif flagged and not args.retry:
        print("re-take them: add --retry --model <model> [--style \"...\"]")


def _qc_retakes(args, cfg: Config, m: Manifest, flagged, checks, final, clip) -> tuple:
    """`qc --retry`: re-take each flagged row in its own voice; updates `final`
    (row id -> Check). -> (take dir, number of bank clips it replaced)."""
    from . import qc
    from .gemini_batch import _write_and_mark
    from .gemini_run import best_of_takes, sync_session
    from .paths import voice_from_path

    instruction, style = _direction_args(args)
    voice_of = {r["id"]: r["voice"] or voice_from_path(r["file_path"]) for r in flagged}
    for v in sorted({v for v in voice_of.values() if v}):
        _check_voice(v, args.model, "--retry")
    take, sent_style, inline = sync_session(model=args.model, instruction=instruction,
                                            style=style, rpm=args.rpm)
    direction = sent_style or inline or None
    take_dir, roots, listing = None, _take_roots(cfg, args.ogg_root), []
    out = pathlib.Path(cfg.out_dir)
    print(f"re-taking {len(flagged)} flagged clip(s) with {args.model}, up to "
          f"{args.qc_retries} take(s) each, {args.rpm:g} rpm")
    for row in flagged:
        word, voice = row["word"], voice_of[row["id"]]
        if not voice:
            print(f"  {word!r}: no voice recorded or in its path, skipped")
            continue
        held = clip(row)
        current = (qc.read_pcm(held), checks[row["id"]])
        pcm, check, made = best_of_takes(word, current, lambda: take(word, voice),
                                         args.qc_retries, print)
        takes = row["qc_takes"] + made
        if pcm is current[0]:               # no re-take beat the current clip
            if row["status"] == RETAKE:     # its re-takes end here: file it
                _write_and_mark(m, out, row["lang"], voice, row, pcm, check=check, qc_takes=takes)
                held.unlink()
            else:
                m.record_qc([(row["id"], check, takes)])
        else:
            if row["status"] != RETAKE:     # a held take never reached the bank
                take_dir = take_dir or _take_dir(cfg)
                _archive_take(cfg, row, take_dir, roots, listing)
            _write_and_mark(m, out, row["lang"], voice, row, pcm, model=args.model,
                            style=direction, check=check, qc_takes=takes)
            if row["status"] == RETAKE:
                held.unlink()               # only once the new take is filed
        final[row["id"]] = check
    return take_dir, len(listing)


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
    qcc = words.qc_counts()
    print(f"words:   {words.counts()}")
    print(f"         QC: {qcc['flagged']} flagged for review, {qcc['unchecked']} never checked, "
          f"{qcc['accepted']} accepted" + (" (python -m wordbank qc)" if qcc["unchecked"] else ""))
    print(f"phrases: {phrases.counts()}")
    words.close()
    phrases.close()


def cmd_export(args, cfg: Config) -> None:
    m = Manifest(cfg.db)
    rows = m.iter_rows()
    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    writer = csv.writer(out)
    cols = ["id", "word", "lang", "voice", "status", "attempts", "file_path", "billed_chars", "error", "model", "style", "review", "qc_seconds", "qc_end_db",
            "qc_takes"]
    writer.writerow(cols)
    for r in rows:
        writer.writerow([r[c] for c in cols])
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

    bs = sub.add_parser("batch-submit",
                        help="submit async job(s) and exit — safe to power off after")
    _add_gemini_args(bs, poll=False)
    bs.set_defaults(func=cmd_batch_submit)

    bc = sub.add_parser("batch-collect",
                        help="poll in-flight job(s) and write clips (run in the morning)")
    _add_gemini_args(bc, submit=False)
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
    gr.add_argument("--voice", default="Kore", type=voice_label, help=_VOICE_HELP)
    gr.add_argument("--model", default=GEMINI_TTS_DEFAULT_MODEL,
                    help=f"Gemini TTS model (default {GEMINI_TTS_DEFAULT_MODEL})")
    _add_direction_args(gr)
    _add_rpm_arg(gr)
    gr.add_argument("--max-attempts", type=int, default=3, dest="max_attempts",
                    help="attempts per word across runs (default 3; QC re-takes don't count)")
    gr.add_argument("--limit", type=int, default=None, help="cap words this run")
    _add_qc_arg(gr)
    gr.set_defaults(func=cmd_gemini_run)

    rg = sub.add_parser("regenerate",
                        help="move the current take of listed words to <db dir>/takes/<UTC>/ "
                             "and reset them to pending")
    rg.add_argument("--words-file", required=True, dest="words_file",
                    help="text file, one word per line (words not in the bank are reported)")
    rg.add_argument("--voice", default=None, type=voice_label,
                    help="only this voice's rows (default: every voice the word is banked in)")
    rg.add_argument("--ogg-root", default=None, dest="ogg_root",
                    help="OGG tree to take the encoded copies from too, e.g. output/words_ogg")
    rg.set_defaults(func=cmd_regenerate)

    q = sub.add_parser("qc",
                       help="check the clips (ends on sound, implausible length), record "
                            "flags; --retry re-takes flagged words until they pass")
    q.add_argument("--voice", default=None, type=voice_label, help="only this voice's clips (default: every voice)")
    q.add_argument("--all-langs", action="store_true", dest="all_langs",
                   help="every language, not just --lang")
    q.add_argument("--csv", default=None,
                   help="write the clips still flagged (id, word, voice, seconds, end_db, "
                        "reason, file_path) here, for listening")
    q.add_argument("--accept-file", default=None, dest="accept_file",
                   help="words (one per line) a person listened to and kept despite a flag: "
                        "their flag is cleared and qc skips them from now on (regenerate undoes it)")
    q.add_argument("--retry", action="store_true",
                   help="re-take flagged words with Gemini (sync, rate-limited), each in its "
                        "own voice, with --model and --style/--instruction")
    q.add_argument("--model", default=GEMINI_TTS_DEFAULT_MODEL,
                   help=f"Gemini TTS model for --retry (default {GEMINI_TTS_DEFAULT_MODEL})")
    _add_direction_args(q)
    _add_rpm_arg(q)
    _add_qc_arg(q)
    q.add_argument("--ogg-root", default=None, dest="ogg_root",
                   help="OGG tree of the bank: a replaced clip's OGG copy is moved aside too, "
                        "so encode_wavs.py encodes the new take")
    q.set_defaults(func=cmd_qc)

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
