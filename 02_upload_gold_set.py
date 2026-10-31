"""02 · The gold set: 195 labelled tool calls, uploaded as a LangSmith dataset with dev/heldout splits.

Needs LANGSMITH_API_KEY. Pass --recreate to replace an existing dataset.
"""

import sys
from collections import Counter

from dotenv import load_dotenv
from langsmith import Client

from gold import DATASET_NAME, load_gold, require_keys, to_langsmith_example

load_dotenv()

rows = load_gold()
print(f"{len(rows)} examples in {len({r['group'] for r in rows})} scenario groups")
for name, counts in [
    ("label", Counter(r["label"] for r in rows)),
    ("split", Counter(r["split"] for r in rows)),
    ("hard", Counter(r["split"] for r in rows if "hard" in r["tags"])),
    ("injection", Counter(r["split"] for r in rows if "injection" in r["tags"])),
]:
    print(f"  {name:<10} {dict(counts)}")

require_keys("LANGSMITH_API_KEY")
client = Client()

if client.has_dataset(dataset_name=DATASET_NAME):
    if "--recreate" not in sys.argv:
        print(f"\ndataset {DATASET_NAME!r} already exists. Pass --recreate to replace it.")
        sys.exit(0)
    client.delete_dataset(dataset_name=DATASET_NAME)

dataset = client.create_dataset(
    DATASET_NAME,
    description="AutoMode gate gold set: tool calls labelled allow / approve / block.",
)
client.create_examples(
    dataset_id=dataset.id, examples=[to_langsmith_example(r) for r in rows]
)
print(f"\nuploaded {len(rows)} examples → {dataset.url}")
