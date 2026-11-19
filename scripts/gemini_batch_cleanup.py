"""List and (optionally) cancel / delete Gemini batch jobs.

The enqueued-token quota counts non-terminal jobs: --cancel stops stuck ones;
deleting finished jobs is tidy-up. Only delete jobs whose results were
collected. Requires GEMINI_API_KEY (the key that created the jobs).

  python scripts/gemini_batch_cleanup.py                 # list jobs + states (no changes)
  python scripts/gemini_batch_cleanup.py --delete        # delete FINISHED wordbank-* jobs
  python scripts/gemini_batch_cleanup.py --delete --cancel  # also cancel + delete running ones
  python scripts/gemini_batch_cleanup.py --delete --all  # ignore the wordbank- prefix filter
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.gemini_batch import is_terminal, job_state
from wordbank.gemini_tts import new_client


def main() -> None:
    p = argparse.ArgumentParser(description="List/delete Gemini batch jobs")
    p.add_argument("--delete", action="store_true", help="delete matching jobs")
    p.add_argument("--cancel", action="store_true",
                   help="cancel non-terminal jobs before deleting (with --delete)")
    p.add_argument("--all", action="store_true",
                   help="don't restrict to the 'wordbank-' display-name prefix")
    p.add_argument("--prefix", default="wordbank-", help="display_name prefix filter")
    args = p.parse_args()

    client = new_client()
    jobs = list(client.batches.list(config={"page_size": 100}))
    if not jobs:
        print("no batch jobs found")
        return

    print(f"{'STATE':<26} {'DISPLAY_NAME':<28} NAME")
    deleted = cancelled = skipped = 0
    for j in jobs:
        st = job_state(j)
        dn = getattr(j, "display_name", "") or ""
        print(f"{st:<26} {dn:<28} {j.name}")
        if not args.delete:
            continue
        if not args.all and not dn.startswith(args.prefix):
            skipped += 1
            continue
        if not is_terminal(j):
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
              "(--cancel to stop running ones).")


if __name__ == "__main__":
    main()
