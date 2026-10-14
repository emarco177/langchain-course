"""01 · One yes/no question. Jev returns a probability.

Needs TYPESAFE_API_KEY for managed Jev, or `ollama pull tev1` for local.
"""

import time

from dotenv import load_dotenv
from langchain_typesafe import Noul, TypeSafeClassifier

load_dotenv()

# Managed Jev:
classifier = TypeSafeClassifier()

# Local Jev on Ollama. Same class, same API.
classifier = TypeSafeClassifier(
    base_url="http://localhost:11434", model="tev1", api_key="ollama"
)


# The key ("urgent") is how you read the answer back.
QUESTIONS = {
    "urgent": Noul(instructions="Does this message need attention right now?"),
}

STATE = "The deploy failed twice and customers are seeing 500s. Can someone look now?"

start = time.perf_counter()
response = classifier.invoke({"state": STATE, "questions": QUESTIONS})
elapsed_ms = (time.perf_counter() - start) * 1000

urgency = response.nouls["urgent"].noul

print(f"model:    {response.model}")
print(f"urgent:   {urgency:.3f}  (probability the answer is yes)")
print(f"latency:  {elapsed_ms:.0f} ms")

# Plain code branches on the number. No parsing.
if urgency >= 0.8:
    print("→ paging the on-call engineer")
else:
    print("→ Open regular Support ticket")
