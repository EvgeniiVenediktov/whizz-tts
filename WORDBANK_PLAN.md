# Word Bank Generation Pipeline — Plan (Cloud TTS)

> **Historical design document.** This records the original decisions and the
> reasoning behind them — it is deliberately *not* kept in sync with the code.
> Several things below were later superseded: the Chirp 3 HD backend was
> removed entirely (replaced by local OmniVoice for accent phrases), Whisper
> verification was dropped, and the CLI has changed. **For current state see
> `AGENTS.md` and `README.md`**; read this for *why* the pipeline is shaped the
> way it is.

Generate a large bank (20k+) of single-word audio clips using a **cloud TTS
backend** (Chirp 3 HD or Gemini TTS), verified locally with Whisper.

## Decisions

| Concern | Decision |
|---------|----------|
| Bulk generation | **Gemini Batch API (decided)** — async, ~50% cheaper, overnight; one word per request, no rpm juggling. Chirp 3 HD parallel-sync kept as alternate |
| Word source | `wordfreq` (aggregates Google Books/Wikipedia/SUBTLEX/OpenSubtitles; 40+ langs). SUBTLEX-US (60,384 words, peer-reviewed) as single-corpus alternative |
| Processed-state | SQLite manifest (`wordbank.db`) is the source of truth — **not** a parallel flat file (see note) |
| Voice | Preset voice (consistent by construction — same named voice every call) |
| Scale | 20k+ words, user-supplied list → SQLite manifest, resumable |
| Language | English now; both backends cover 50+ languages for later |
| Mapping | **Keyed 1:1** — file-based JSONL, `key`=manifest id; results matched by key, never by position |
| Receive format | **WAV** (raw PCM wrapped, no ffmpeg/trim/normalize at receive). Optional OGG/Opus later via `scripts/encode_wavs.py` |
| Verification | **None** (Whisper dropped). Correctness comes from keyed mapping; `word` column is the text↔recording index |

## Processed-state design (why a SQLite manifest, not a flat `done.txt`)

`words.txt` is the **immutable input**. Tracking which words are done lives in
the SQLite manifest, not a parallel text file, because:
- a flat `done.txt` appended from parallel workers races on concurrent writes;
- it can only record "done", not failed / quarantined / submitted / attempts /
  error / which batch a word is in — all of which we need for resume;
- a crash mid-append can corrupt a line.
The manifest gives atomic per-word state, dedupe on ingest, and resumable batch
tracking (`batch_id`). A human-readable snapshot is produced on demand via
`export` (CSV), not kept as live state.

## Implementation status *(as of the original plan — since superseded)*

`wordbank/` — `manifest.py` (SQLite, thread-safe, `submitted`+`batch_id`),
`gemini_batch.py` (submit→poll→decode→OGG, resumable), `backends/chirp.py`
(sync alternate), `pipeline.py` (sync + `run_parallel`), `ratelimit.py`,
`verify.py` (Whisper), `postproc.py`, `wordlist.py`, CLI (`build` / `batch` /
`status` / `export`). `scripts/fetch_wordlist.py` (wordfreq) and
`scripts/gemini_batch_smoke.py` (verify-before-bulk).

Since removed: `backends/chirp.py` (Chirp dropped for local OmniVoice) and
`pipeline.run_parallel` (dead — the only wired backend isn't thread-safe).
See `AGENTS.md` → *Key files* for the current layout.

⚠️ Two things to confirm with `scripts/gemini_batch_smoke.py` on ~10 words +
a real key before a 20k run: (1) batch mode accepts the per-request TTS config
on the chosen model; (2) for ~40k+ requests switch inline `src` → JSONL file
with explicit keys (inline ~20MB limit; files 2GB).

## Why this is simpler than the local pipeline

Cloud models render an isolated word reliably, so **one request = one word = one
file**. That deletes the entire batch → word-timestamp → cut → align stage that
OmniVoice required. The only real per-word risk is **homographs** ("read",
"lead", "bass"), handled by SSML phoneme overrides + the Whisper check.

## Verified facts (sources at bottom)

- **Chirp 3 HD:** GA, 28 voices, 54 languages, supports SSML `<phoneme>`
  (IPA/X-SAMPA) for pronunciation control + `[pause]` markup. Character-priced.
- **Gemini 2.5 Flash/Pro TTS:** GA on Vertex (Preview on Dev API). Audio = 25
  tokens/sec. Flash $10/1M output tokens ($5 Batch); Pro $20 ($10 Batch). 30
  voices, 100+ languages. Preset voices only.
- **Cloud TTS free tier:** 4M Standard + 1M WaveNet chars/month free (Chirp 3's
  exact allotment/rate unconfirmed — verify on the pricing page).

## Cost (20k words ≈ 120k chars; ~0.9s/word audio)

| Backend | Basis | Est. cost | Note |
|---------|-------|-----------|------|
| Chirp 3 HD | ~120k chars | **~$0–4** | likely within/near free tier; confirm rate |
| Gemini Flash TTS | ~0.45M audio tok | **~$4.5** (std) / **~$2.3** (Batch) | + ~25% retries |
| Gemini Pro TTS | ~0.45M audio tok | **~$9** (std) / **~$4.5** (Batch) | higher quality |

Cost is a non-issue at this scale. Track real usage per word in the manifest.

## Architecture

```
words.txt ─► [Manifest: SQLite]  ◄── resumable, dedup, per-word status + usage
                  │  pull pending words
                  ▼
        [TTS Backend]  (pluggable)
           ├─ ChirpTTSBackend   → Cloud TTS, voice en-US-Chirp3-HD-<name>, OGG_OPUS
           └─ GeminiTTSBackend  → Gemini generateContent AUDIO → PCM → OGG (src/audio.py)
                  │  optional SSML <phoneme> for known homographs
                  ▼  (concurrent, rate-limited thread pool)
        [Post-proc]  trim trailing silence · loudness-normalize · OGG/Opus
                  │
                  ▼  local Whisper STT, normalized match (digit/phonetic tolerant)
        [Verifier]  pass ─► write file + mark verified
                  │ fail
                  ▼  retry (SSML phoneme override) → quarantine + log
```

## Components

### 1. Manifest (SQLite) — `wordbank/manifest.py`
`id, word, lang, backend, voice, status, attempts, file_path, transcript,
chars_or_tokens, error, created_at`. Resume = `status IN ('pending','failed')
AND attempts < MAX`.

### 2. Backend interface — `wordbank/backends/`
```
class TTSBackend:
    def synthesize(self, text, *, voice, language, ssml=None) -> bytes  # OGG/Opus PCM-or-encoded
```
- **ChirpTTSBackend**: `google-cloud-texttospeech`; `audioEncoding=OGG_OPUS`
  (fallback LINEAR16 → `src/audio.py` if a voice restricts encoding).
- **GeminiTTSBackend**: `google-genai`, `response_modalities=["AUDIO"]`, voice
  via `SpeechConfig`; returns PCM → `pcm_to_ogg`.

### 3. Pronunciation overrides — `wordbank/lexicon.py`
Curated `{word: IPA}` map for known-ambiguous homographs → emitted as
`<phoneme>` SSML (Chirp 3) or pronunciation hint in the Gemini prompt.
Seeded from the words Whisper flags as mismatches.

### 4. Verification — `wordbank/verify.py`
Reuse loaded Whisper. Normalize: lowercase, strip punct, number-word map,
metaphone fallback. Duration sanity (non-empty, not absurd).

### 5. Driver — `wordbank/pipeline.py`
Concurrent workers (network-bound) with a rate limiter + exponential backoff on
429/5xx. Per-word retry ladder: plain → SSML phoneme override → quarantine (logged).

### 6. Output layout
```
output/words/<lang>/<backend>-<voice>/<id // 1000>/<id>_<word>.ogg
wordbank.db
```

### 7. CLI — `python -m wordbank …`
`build --words words.txt --backend chirp|gemini --voice <name> --lang en
--concurrency 20` · `retry-failed` · `status` · `export` · `estimate-cost`.

## Config (.env)
- `GOOGLE_APPLICATION_CREDENTIALS` (Chirp / Cloud TTS service account)
- `GEMINI_API_KEY` (Gemini backend)
- `WORDBANK_BACKEND`, `WORDBANK_VOICE`, `WORDBANK_CONCURRENCY`

## Risks & mitigations
| Risk | Mitigation |
|------|-----------|
| Homographs mispronounced | curated IPA `<phoneme>` overrides; Whisper flags them |
| Rate limits / quotas (esp. Gemini Preview) | concurrency cap + backoff; Gemini Batch API |
| Voice restricts OGG encoding | request LINEAR16, encode via `src/audio.py` |
| Whisper false-rejects (homophones) | digit + phonetic normalization; manual quarantine review |
| Cloud dependency / data leaves infra | accepted; documented. Offline fallback below |
| Chirp 3 exact price unconfirmed | verify pricing page before bulk run; `estimate-cost` first |

## Phases
1. **MVP:** manifest + ChirpTTSBackend + one-word-per-request + OGG + Whisper
   verify + resume. English, one voice.
2. **Robustness:** GeminiTTSBackend, SSML phoneme overrides, retry/quarantine,
   loudness normalization, `status`/`export`/`estimate-cost` CLIs, backoff.
3. **Scale & i18n:** tuned concurrency / Gemini Batch, multilingual word lists +
   per-language voice/phoneme/STT-hint.

## Backend (decided): Chirp 3 HD
Chosen for: GA, deterministic, IPA `<phoneme>` pronunciation control, and
character-priced (likely ~free at this scale). Gemini TTS remains implementable
behind the same `TTSBackend` interface if its language breadth/style is wanted later.

**Credentials note:** Chirp 3 HD is **Cloud Text-to-Speech**, not the Gemini
Developer API. It needs a **GCP project with the Text-to-Speech API enabled** and
a service-account key (`GOOGLE_APPLICATION_CREDENTIALS`) — the `ai.google.dev`
Gemini key does **not** authenticate Chirp 3 HD.

---

## Appendix — offline / cloned-voice fallback
If a requirement later changes to **your own cloned voice** or **fully offline**
generation, use the local OmniVoice batch→word-timestamp→cut→verify pipeline
(validated in `scripts/validate_batch_words.py`; see git history of this file).

## Sources
- Gemini TTS pricing & 25 tok/s: https://ai.google.dev/gemini-api/docs/pricing
- Gemini speech generation: https://ai.google.dev/gemini-api/docs/speech-generation
- Gemini-TTS (Cloud, GA/preset): https://docs.cloud.google.com/text-to-speech/docs/gemini-tts
- Chirp 3 HD (GA, SSML/phoneme): https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
- Cloud TTS pricing & free tier: https://cloud.google.com/text-to-speech/pricing
