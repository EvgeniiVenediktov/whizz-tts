"""List and (optionally) delete Gemini batch jobs to free the enqueued-token quota.

Finished jobs can keep counting against the batch enqueued-token quota until they
are deleted (or auto-expire ~48h after creation). Once you've collected results,
delete them to free the budget immediately.

Requires GEMINI_API_KEY (the SAME key used to create the jobs).

  python scripts/gemini_batch_cleanup.py                 # list jobs + states (no changes)
  python scripts/gemini_batch_cleanup.py --delete        # delete FINISHED wordbank-* jobs
  python scripts/gemini_batch_cleanup.py --delete --cancel  # also cancel + delete running ones
  python scripts/gemini_batch_cleanup.py --delete --all  # ignore the wordbank- prefix filter
"""
import argparse

from google import genai

_TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")


def _state(job) -> str:
    s = getattr(job, "state", "")
    return getattr(s, "name", None) or str(s)


def main() -> None:
    p = argparse.ArgumentParser(description="List/delete Gemini batch jobs")
    p.add_argument("--delete", action="store_true", help="delete matching jobs")
    p.add_argument("--cancel", action="store_true",
                   help="cancel non-terminal jobs before deleting (with --delete)")
    p.add_argument("--all", action="store_true",
                   help="don't restrict to the 'wordbank-' display-name prefix")
    p.add_argument("--prefix", default="wordbank-", help="display_name prefix filter")
    args = p.parse_args()

    client = genai.Client()
    jobs = list(client.batches.list(config={"page_size": 100}))
    if not jobs:
        print("no batch jobs found")
        return

    print(f"{'STATE':<26} {'DISPLAY_NAME':<28} NAME")
    deleted = cancelled = skipped = 0
    for j in jobs:
        st = _state(j)
        dn = getattr(j, "display_name", "") or ""
        print(f"{st:<26} {dn:<28} {j.name}")
        if not args.delete:
            continue
        if not args.all and not dn.startswith(args.prefix):
            skipped += 1
            continue
        terminal = st.endswith(_TERMINAL)
        if not terminal:
            if args.cancel:
                client.batches.cancel(name=j.name)
                cancelled += 1
            else:
                print(f"  ↳ skip (still {st}; pass --cancel to stop it)")
                skipped += 1
                continue
        client.batches.delete(name=j.name)
        deleted += 1

    if args.delete:
        print(f"\ndeleted {deleted}, cancelled {cancelled}, skipped {skipped}")
    else:
        print(f"\n{len(jobs)} job(s). Re-run with --delete to remove finished ones "
              "and free the enqueued-token quota.")


if __name__ == "__main__":
    main()
