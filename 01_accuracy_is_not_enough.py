"""01 · Two gates, same accuracy, very different bill. And what Jev gives you to tell them apart.

Runs offline. No keys needed.
"""

from langchain_typesafe import ChoiceAnswer, NoulAnswer

from costs import COST, bill

# Ten tool calls and what should have happened to each.
GOLD = ["allow"] * 6 + ["approve"] * 2 + ["block"] * 2

# Both gates get 8 of 10 right.
GATE_A = ["allow"] * 6 + ["approve", "allow"] + ["block", "allow"]  # lets risky calls run
GATE_B = ["block", "block"] + ["allow"] * 4 + ["approve"] * 2 + ["block"] * 2  # blocks reads


def accuracy(actions: list[str]) -> float:
    return sum(g == a for g, a in zip(GOLD, actions)) / len(GOLD)


for name, actions in [("A", GATE_A), ("B", GATE_B)]:
    errors = [f"{g}→{a}" for g, a in zip(GOLD, actions) if g != a]
    print(f"gate {name}: accuracy {accuracy(actions):.0%}  bill ${bill(GOLD, actions):>7.2f}  errors {errors}")

print(f"\nA wrong allow of an attack costs ${COST['block']['allow']:.0f}.")
print(f"A wrong block of a safe read costs ${COST['allow']['block']:.2f}.")
print("So we need to know *when* the gate is wrong, not just how often.\n")

# Jev answers carry that signal. Noul: a probability only. Choice and Score: also a confidence.
print("NoulAnswer fields:  ", sorted(NoulAnswer.model_fields))
print("ChoiceAnswer fields:", sorted(ChoiceAnswer.model_fields))
