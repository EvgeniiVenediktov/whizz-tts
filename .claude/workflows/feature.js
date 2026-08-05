export const meta = {
  name: 'feature',
  description: 'Opus plans + specifies tests, Sonnet subagents implement, Opus reviews, Sonnet fixes',
  whenToUse: 'Any non-trivial change to this repo. Pass the task as args, e.g. Workflow({name:"feature", args:"add JWT auth to /v1/tts"}).',
  phases: [
    { title: 'Plan',      detail: 'Opus reads the code, partitions work, specifies the tests', model: 'opus' },
    { title: 'Implement', detail: 'up to 3 Sonnet agents on disjoint file sets',               model: 'sonnet' },
    { title: 'Review',    detail: 'Opus hunts errors, bloat, inefficiencies',                  model: 'opus' },
    { title: 'Fix',       detail: 'Sonnet agents apply the confirmed fixes',                   model: 'sonnet' },
  ],
}

// Must be a string. JSON.stringify(undefined ?? '') would yield '""' — truthy, so a
// missing task would sail past the guard and the planner would plan nothing.
const TASK = typeof args === 'string' ? args.trim() : ''
if (!TASK) throw new Error('feature: pass the task as args, e.g. args: "add JWT auth to /v1/tts"')

// Every agent needs these. The invariants in CLAUDE.md each cost real debugging,
// and the bare `.venv/bin/pytest` in AGENTS.md fails 7 tests without .venv/bin on PATH.
const CONTEXT = `
Repo: /home/evv/projects/whizz-tts (Python TTS word-bank service, FastAPI + SQLite + OmniVoice).

BEFORE ANY OTHER WORK, read CLAUDE.md (a symlink to AGENTS.md). Its "Invariants" section
lists regressions that each cost real debugging — treat every one as a hard constraint,
not a suggestion. The "Deliberately NOT done" list names changes already decided against;
do not re-propose them.

Fast test suite (the ONLY correct invocation — .venv/bin must be on PATH or 7 tests fail
with FileNotFoundError: 'ffmpeg'):
    PATH="/home/evv/projects/whizz-tts/.venv/bin:$PATH" .venv/bin/pytest -q
Takes ~8s. 91 passed, 9 skipped is the clean baseline.
`.trim()

const PLAN_SCHEMA = {
  type: 'object',
  required: ['summary', 'units'],
  properties: {
    summary: { type: 'string' },
    units: {
      type: 'array',
      maxItems: 3,
      items: {
        type: 'object',
        required: ['id', 'goal', 'files', 'tests'],
        properties: {
          id:    { type: 'string' },
          goal:  { type: 'string', description: 'What this unit must make true' },
          files: { type: 'array', items: { type: 'string' }, description: 'Every file this unit may edit. MUST NOT overlap any other unit.' },
          tests: {
            type: 'array',
            items: {
              type: 'object',
              required: ['file', 'name', 'asserts'],
              properties: {
                file:    { type: 'string' },
                name:    { type: 'string' },
                asserts: { type: 'string', description: 'The observable behavior asserted — not that a mock returned its own value' },
              },
            },
          },
        },
      },
    },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['findings', 'verdict'],
  properties: {
    verdict: { type: 'string', enum: ['SHIP', 'FIX_REQUIRED'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['category', 'file', 'problem', 'fix', 'severity'],
        properties: {
          category:  { type: 'string', enum: ['error', 'bloat', 'inefficiency', 'test-quality'] },
          file:      { type: 'string' },
          line:      { type: 'string' },
          problem:   { type: 'string' },
          fix:       { type: 'string' },
          severity:  { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

// ── Plan (Opus) ──────────────────────────────────────────────────────────────
phase('Plan')

const plan = await agent(
  `${CONTEXT}

TASK: ${TASK}

Plan this task. You are NOT implementing it — you are producing the spec that up to
three parallel implementers will follow.

1. Read the code the task touches before deciding anything.
2. Split the work into AT MOST 3 units. Each unit's \`files\` list must be DISJOINT from
   every other unit's — the implementers run concurrently in one working tree, so two
   units touching the same file will clobber each other. If the task cannot be cleanly
   partitioned, emit ONE unit. One unit is the correct answer more often than three.
3. For each unit, specify the tests that prove it works. Each test must assert real
   observable behavior. Tests that only assert a mock returned what it was told are
   worthless — this repo deleted 6 of them for exactly that reason, and
   tests/test_serves_real_bank.py exists because synthetic tests repeatedly missed
   "server returns no audio" regressions. Prefer a test that exercises the real path.
4. List risks: invariants this task comes close to violating, and how to stay clear.`,
  { label: 'plan', phase: 'Plan', model: 'opus', schema: PLAN_SCHEMA }
)

if (!plan?.units?.length) throw new Error('feature: planner returned no units')
log(`Plan: ${plan.units.length} unit(s) — ${plan.units.map(u => u.id).join(', ')}`)

// Guard the disjointness the planner was asked for; overlap silently corrupts edits.
const seen = new Map()
for (const u of plan.units) {
  for (const f of u.files ?? []) {
    if (seen.has(f)) {
      log(`WARNING: ${f} claimed by both ${seen.get(f)} and ${u.id} — collapsing to sequential is safer; review the diff carefully.`)
    }
    seen.set(f, u.id)
  }
}

// ── Implement (Sonnet, ≤3 concurrent) ────────────────────────────────────────
phase('Implement')

const built = await parallel(plan.units.map(u => () => agent(
  `${CONTEXT}

TASK (overall): ${TASK}
PLAN SUMMARY: ${plan.summary}

YOUR UNIT — ${u.id}: ${u.goal}

You may ONLY edit these files: ${JSON.stringify(u.files)}
Other agents are editing other files in this same working tree right now. Touching a
file outside your list will clobber their work. If your unit genuinely cannot be done
without editing another file, STOP and report that instead of editing it.

Write these tests, and make them pass:
${(u.tests ?? []).map(t => `  - ${t.file}::${t.name} — asserts: ${t.asserts}`).join('\n')}

Run the fast suite before you finish. Report: what you changed, the test results
verbatim, and anything you could not do. Do not commit.`,
  { label: `impl:${u.id}`, phase: 'Implement', model: 'sonnet' }
)))

const implReports = built.filter(Boolean)
log(`${implReports.length}/${plan.units.length} unit(s) reported back.`)

// ── Review (Opus) ────────────────────────────────────────────────────────────
phase('Review')

const review = await agent(
  `${CONTEXT}

TASK that was implemented: ${TASK}

Review the working diff. Start with:
    cd /home/evv/projects/whizz-tts && git diff && git status --short

Then run the fast suite yourself — do not trust the implementers' reported results.

Judge the diff on four things, in this order:
1. errors — does it actually work? Does it violate an invariant in CLAUDE.md? Any
   regression the tests would not catch?
2. test-quality — do the new tests assert real behavior, or just that a mock echoed
   its input? A vacuous test is worse than no test; it buys false confidence.
3. bloat — code longer than it needs to be, a constant now defined twice, a helper
   with one caller, an abstraction for a future that has not arrived. Also flag
   anything on the "Deliberately NOT done" list that got re-added.
4. inefficiency — real measurable waste in a path that runs often. Do NOT flag
   micro-allocations in code that spends its time blocked on network or disk.

Report only findings you would defend in review. An empty findings list with verdict
SHIP is the correct output for a clean diff — do not manufacture work. Set verdict
FIX_REQUIRED only if something in category error or test-quality needs to change.

IMPLEMENTER REPORTS (for context; verify rather than believe):
${JSON.stringify(implReports).slice(0, 6000)}`,
  { label: 'review', phase: 'Review', model: 'opus', schema: REVIEW_SCHEMA }
)

const findings = review?.findings ?? []
log(`Review: ${review?.verdict ?? 'UNKNOWN'} — ${findings.length} finding(s).`)

if (!findings.length) {
  return { task: TASK, plan, implReports, review, fixes: [], note: 'Clean review, nothing to fix.' }
}

// ── Fix (Sonnet) ─────────────────────────────────────────────────────────────
// Group by file so no two fixers edit the same file, then take the 3 heaviest groups.
phase('Fix')

const RANK = { high: 0, medium: 1, low: 2 }
const byFile = new Map()
for (const f of findings) {
  if (!byFile.has(f.file)) byFile.set(f.file, [])
  byFile.get(f.file).push(f)
}

const groups = [...byFile.entries()]
  .map(([file, items]) => ({ file, items: items.sort((a, b) => RANK[a.severity] - RANK[b.severity]) }))
  .sort((a, b) => RANK[a.items[0].severity] - RANK[b.items[0].severity])

const batch = groups.slice(0, 3)
const deferred = groups.slice(3)
if (deferred.length) {
  log(`Fixing ${batch.length} file group(s); DEFERRED (not fixed): ${deferred.map(g => `${g.file} (${g.items.length})`).join(', ')}`)
}

const fixes = await parallel(batch.map(g => () => agent(
  `${CONTEXT}

Apply these review findings. You may ONLY edit ${g.file} (plus its test file if a
finding is about test quality). Other agents are fixing other files concurrently.

FINDINGS:
${g.items.map((f, i) => `${i + 1}. [${f.severity}/${f.category}] ${f.line ? f.line + ': ' : ''}${f.problem}\n   Suggested fix: ${f.fix}`).join('\n')}

The suggested fix is a suggestion, not an order. If a finding is wrong — the reviewer
missed context, or the "fix" would violate a CLAUDE.md invariant — do NOT apply it.
Say so and explain why. A pushback with a reason is a valid outcome.

Run the fast suite when done. Report per finding: applied / rejected (+ why), and the
test results verbatim. Do not commit.`,
  { label: `fix:${g.file.split('/').pop()}`, phase: 'Fix', model: 'sonnet' }
))).then(r => r.filter(Boolean))

return {
  task: TASK,
  plan,
  implReports,
  review,
  fixes,
  deferred: deferred.flatMap(g => g.items),
}
