export const meta = {
  name: 'tts-review',
  description: 'Multi-dimension automated code review of the TTS MVP codebase',
  phases: [
    { title: 'Triage', detail: '4 parallel auditors: dead-code, bloat, token-opt, deployment' },
    { title: 'Judge',  detail: 'Independent architect assessment' },
    { title: 'Verify', detail: 'Adversarial finding verification' },
    { title: 'Synthesize', detail: 'Final prioritized punch list' },
  ],
}

// ── Schemas ──────────────────────────────────────────────────────────────────

const FINDING = {
  type: 'object',
  required: ['dimension', 'file', 'description', 'severity', 'fix'],
  properties: {
    dimension:   { type: 'string' },
    file:        { type: 'string' },
    line_range:  { type: 'string' },
    description: { type: 'string' },
    severity:    { type: 'string', enum: ['high', 'medium', 'low'] },
    fix:         { type: 'string' },
  },
}

const FINDINGS_SCHEMA = {
  type: 'object',
  required: ['findings'],
  properties: { findings: { type: 'array', items: FINDING } },
}

const VERIFIED_FINDING = {
  type: 'object',
  required: ['dimension', 'file', 'description', 'severity', 'fix', 'verdict'],
  properties: {
    dimension:      { type: 'string' },
    file:           { type: 'string' },
    line_range:     { type: 'string' },
    description:    { type: 'string' },
    severity:       { type: 'string', enum: ['high', 'medium', 'low'] },
    fix:            { type: 'string' },
    verdict:        { type: 'string', enum: ['CONFIRMED', 'DISPUTED'] },
    dispute_reason: { type: 'string' },
  },
}

const VERIFIED_SCHEMA = {
  type: 'object',
  required: ['findings'],
  properties: { findings: { type: 'array', items: VERIFIED_FINDING } },
}

const JUDGE_SCHEMA = {
  type: 'object',
  required: ['architect_summary'],
  properties: { architect_summary: { type: 'array', items: { type: 'string' } } },
}

const REPORT_SCHEMA = {
  type: 'object',
  required: ['findings_by_dimension', 'architect_summary', 'top_10'],
  properties: {
    findings_by_dimension: {
      type: 'object',
      properties: {
        dead_code:          { type: 'array', items: VERIFIED_FINDING },
        bloat:              { type: 'array', items: VERIFIED_FINDING },
        token_optimization: { type: 'array', items: VERIFIED_FINDING },
        deployment:         { type: 'array', items: VERIFIED_FINDING },
      },
    },
    architect_summary: { type: 'array', items: { type: 'string' } },
    top_10: {
      type: 'array',
      items: {
        type: 'object',
        required: ['rank', 'dimension', 'file', 'description', 'severity', 'fix'],
        properties: {
          rank:        { type: 'number' },
          dimension:   { type: 'string' },
          file:        { type: 'string' },
          line_range:  { type: 'string' },
          description: { type: 'string' },
          severity:    { type: 'string', enum: ['high', 'medium', 'low'] },
          fix:         { type: 'string' },
        },
      },
    },
  },
}

// ── Phase 1: Triage (parallel) ───────────────────────────────────────────────

phase('Triage')
log('Running 4 parallel triage auditors...')

const [deadCode, bloat, tokenOpt, deployment] = await parallel([

  () => agent(
    `You are a dead-code auditor for a Python TTS codebase at /home/evv/projects/whizz-tts.
Read these files: wordbank/__main__.py, wordbank/pipeline.py, wordbank/ratelimit.py,
wordbank/backends/local.py, wordbank/backends/base.py, requirements.txt,
requirements-base.txt, requirements-model.txt, scripts/gemini_sync.py,
scripts/validate_batch_words.py, scripts/diagnose_voice_design.py.

Find code, imports, or dependencies no longer used or superseded. Specifically:
1. Python packages in requirements*.txt never imported in wordbank/ or src/ (grep
   all .py files before concluding dead — e.g. google-cloud-texttospeech).
2. Functions defined but never called from __main__.py, scripts/, or tests/ (check
   run_parallel, RateLimiter).
3. Scripts that test/validate an approach abandoned before the current architecture.
4. Any remnants of a previous Chirp or Google Cloud TTS backend.
5. Instance attributes assigned but never read (e.g. self.language_code vs self.lang).

Severity: high=broken at runtime or security risk; medium=bloats install/confuses
contributors; low=cosmetic clutter.

dimension value must be: "dead_code"`,
    { label: 'dead-code', phase: 'Triage', schema: FINDINGS_SCHEMA }
  ),

  () => agent(
    `You are a simplification auditor for a Python TTS MVP codebase at
/home/evv/projects/whizz-tts. Read these files: wordbank/gemini_batch.py,
wordbank/gemini_tts.py, wordbank/backends/local.py, wordbank/pipeline.py,
wordbank/postproc.py, wordbank/manifest.py, src/server.py, src/audio.py.

Find code longer than necessary, duplicated, or over-engineered. Specifically:
1. Constants defined in more than one file — list all definition sites (SAMPLE_RATE
   appears in gemini_tts.py, gemini_batch.py, backends/local.py, src/audio.py —
   count every occurrence).
2. Functions with the same name but different signatures across files (write_wav in
   gemini_batch.py bytes-variant vs postproc.py ndarray-variant).
3. Instance attributes that alias each other with no functional difference.
4. Stale phase/TODO comments that no longer reflect project state.
5. Methods on classes that are architecturally dead-ends (defined for a future that
   hasn't arrived, on a backend documented as not thread-safe).
6. Model version strings hardcoded in multiple files instead of one constant.

Severity: high=causes confusion or runtime inconsistency; medium=unnecessary
cognitive load; low=minor style issue.

dimension value must be: "bloat"`,
    { label: 'bloat', phase: 'Triage', schema: FINDINGS_SCHEMA }
  ),

  () => agent(
    `You are an LLM cost-optimization auditor. This codebase makes paid Gemini API
calls. Read these files: wordbank/gemini_batch.py, wordbank/gemini_tts.py,
src/server.py, wordbank/__main__.py.

Find patterns that waste tokens or inflate Gemini API cost. Specifically:
1. VOICE_INSTRUCTION in gemini_batch.py — measure its character length, compute
   extra characters added per 2000-word batch job, assess if a shorter instruction
   serves the same goal. Is it configurable or hardcoded everywhere it is used?
2. build_request_body — does it construct a new dict from scratch per word, or
   reuse a template? Identify the minimum structure Gemini Batch API requires.
3. genai.Client() instantiation in gemini_batch.py — created once per process or
   per call? Same question for synthesize_pcm in gemini_tts.py.
4. Any large data payloads passed through Python objects unnecessarily before
   JSON serialization.
5. Inconsistency in instruction format between single-phrase and batch synthesis
   paths — could they share one instruction module?

Severity: high=measurable cost impact per run; medium=inefficiency; low=minor.

dimension value must be: "token_optimization"`,
    { label: 'token-opt', phase: 'Triage', schema: FINDINGS_SCHEMA }
  ),

  () => agent(
    `You are a deployment and architecture auditor. Read these files: Dockerfile,
docker-compose.yml, docker-entrypoint.sh, scripts/pack.sh, src/server.py.

Find architectural problems, missing reliability patterns, security gaps. Specifically:
1. HEALTHCHECK — is there a HEALTHCHECK instruction in Dockerfile? A healthcheck
   key in docker-compose.yml? The /health endpoint exists — is it wired?
2. CORS — what are allow_origins/methods/headers? Any auth? Who can POST /v1/voices
   (writes WAV to disk) with no credentials?
3. BANK_SYNC_CMD in docker-entrypoint.sh — how invoked? Sanitized? Injection risk?
4. asyncio.get_event_loop() inside async def (deprecated Python 3.10+; should be
   asyncio.get_running_loop()) — find all occurrences in server.py.
5. save_voice endpoint — does it call app.state.bank.close() mid-flight? Race
   window for concurrent requests? Is this guarded by a lock?
6. pack.sh — does it bundle source code already in the Docker image? Consequence?
7. Entrypoint failure mode — what happens if verify fails AND no archive exists?
   Does server start against an empty bank? Restart policy? Memory limits?

Severity: high=exploitable or causes data loss; medium=reliability gap;
low=operational inconvenience.

dimension value must be: "deployment"`,
    { label: 'deployment', phase: 'Triage', schema: FINDINGS_SCHEMA }
  ),

])

// ── Phase 2: Architect Judge ──────────────────────────────────────────────────

phase('Judge')
log('Architect judge reading codebase independently...')

const judgeResult = await agent(
  `You are a senior distributed-systems architect reviewing a Python TTS word-bank
service at /home/evv/projects/whizz-tts for production readiness. Read these files
INDEPENDENTLY — do not assume any prior review findings: src/server.py,
wordbank/manifest.py, wordbank/lookup.py, wordbank/gemini_batch.py, Dockerfile,
docker-compose.yml, docker-entrypoint.sh, README.md.

Assess across five axes:
1. Single points of failure — SQLite as sole state store (lock, corruption, volume
   provisioning race on cold start).
2. Reliability patterns — HEALTHCHECK wiring, restart policies, in-flight request
   draining on SIGTERM.
3. Security posture — authentication model, who can write files to disk, whether
   GEMINI_API_KEY is exposed in env, BANK_SYNC_CMD injection surface.
4. Scalability limits — SQLite reader concurrency model, asyncio thread-pool for
   CPU-bound synthesis, horizontal scaling constraints.
5. Operational gaps — how an operator learns the bank is partial or corrupt; Gemini
   quota exhaustion experience; absence of metrics/tracing surface.

Be specific and reference file+line where relevant. Each bullet should be one
complete, actionable observation.`,
  { label: 'architect-judge', phase: 'Judge', schema: JUDGE_SCHEMA }
)

// ── Phase 3: Verify ──────────────────────────────────────────────────────────

phase('Verify')

const allFindings = [
  ...(deadCode?.findings   ?? []),
  ...(bloat?.findings      ?? []),
  ...(tokenOpt?.findings   ?? []),
  ...(deployment?.findings ?? []),
].filter(Boolean)

log(`Adversarially verifying ${allFindings.length} raw findings...`)

const verified = await agent(
  `You are an adversarial code reviewer. Challenge each finding below and mark it
CONFIRMED or DISPUTED with a reason.

Known legitimate patterns that must NOT be flagged as problems:
- BankLookup uses a normal (not mode=ro) SQLite connection + PRAGMA query_only.
  This is intentional for WAL read visibility (documented in AGENTS.md). Do not
  flag as insecure or wrong.
- src.model and src.audio are imported lazily inside request handlers. Intentional
  for bank-only Docker builds (zero ML deps in lightweight image). Do not flag as
  unused imports.
- wordbank/manifest.py uses threading.Lock() guarding all DB writes. Do not flag
  set_status calls as racy without first confirming the lock is absent.
- run_parallel on WordBankPipeline — verify it has zero callers across the entire
  repo before marking dead. If tests or scripts call it, it is not dead.

For each finding: add "verdict": "CONFIRMED" or "DISPUTED", and "dispute_reason"
explaining why if DISPUTED. Preserve all other fields unchanged.

FINDINGS:
${JSON.stringify(allFindings)}`,
  { label: 'verifier', phase: 'Verify', schema: VERIFIED_SCHEMA }
)

const confirmed = (verified?.findings ?? []).filter(f => f.verdict === 'CONFIRMED')
log(`${confirmed.length} of ${allFindings.length} findings confirmed.`)

// ── Phase 4: Synthesize ───────────────────────────────────────────────────────

phase('Synthesize')
log('Generating prioritized punch list...')

const report = await agent(
  `You are a technical lead producing a final action plan from a code review.
You receive CONFIRMED findings (post adversarial verification) and an independent
architect assessment.

Produce:
- findings_by_dimension: group confirmed findings by dimension key, sorted
  high → medium → low severity within each group.
- architect_summary: pass through the architect bullet list unchanged.
- top_10: the 10 highest-priority items across all dimensions.

Prioritization order for top_10 (rank 1 = most urgent):
1. Broken imports causing ImportError at runtime.
2. Unauthenticated write endpoints (file writes with wildcard CORS, no auth).
3. Command injection risks (BANK_SYNC_CMD via sh -c).
4. Missing container health wiring (HEALTHCHECK absent from Dockerfile/compose).
5. Race conditions in the serving path (bank close mid-flight).
6. Deprecated asyncio APIs (get_event_loop in Python 3.10+).
7. Dead dependencies that bloat the install or production image.
8. Quantified token/cost waste (include magnitude in description).
9. Code quality and simplification findings.

CONFIRMED_FINDINGS:
${JSON.stringify(confirmed)}

ARCHITECT_SUMMARY:
${JSON.stringify(judgeResult)}`,
  { label: 'punch-list', phase: 'Synthesize', schema: REPORT_SCHEMA }
)

return report
