"""01 · One yes/no question. Jev returns a probability, not text.

Needs TYPESAFE_API_KEY.
"""

import time

from dotenv import load_dotenv
from langchain_typesafe import Noul, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier()

# The key ("urgent") is how you read the answer back.
QUESTIONS = {
    "urgent": Noul(instructions="Does this message need attention right now?"),
}

STATE = "The deploy failed twice and customers are seeing 500s. Can someone look now?"

start = time.perf_counter()
response = classifier.invoke({"state": STATE, "questions": QUESTIONS})
elapsed_ms = (time.perf_counter() - start) * 1000

urgency = response.nouls["urgent"].noul

print(f"state:      {STATE!r}")
print(f"urgent:     {urgency:.3f}  (probability the answer is yes)")
print(
    f"usage:      {response.usage.input_tokens} in / {response.usage.output_tokens} out"
)
print(f"latency:    {elapsed_ms:.0f} ms")

# Plain code branches on the number. No parsing.
if urgency >= 0.8:
    print("→ paging the on-call engineer")
else:
    print("→ filing in the normal queue")
