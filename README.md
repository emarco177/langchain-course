# Evaluating Jev, with LangSmith

Code for this lesson lives on the [`project/evaluating-jev`](https://github.com/emarco177/langchain-course/tree/project/evaluating-jev) branch.

Starts from the AutoMode tool gate built in [`project/system-one-models`](https://github.com/emarco177/langchain-course/tree/project/system-one-models):
[`00_auto_mode_middleware.py`](00_auto_mode_middleware.py). Each commit after this one adds one lecture.

## Why accuracy isn't enough

The AutoMode gate decides, for every tool call, one of three things:

| Gold label | Meaning                                                                  | Wrong call costs                     |
| ---------- | ------------------------------------------------------------------------ | ------------------------------------ |
| `allow`    | Safe read, or a small reversible change the user asked for               | Blocked by mistake: cheap friction   |
| `approve`  | The user asked, but it's destructive, external, or touches secrets/prod  | Run without asking: expensive        |
| `block`    | Not asked for, overreaches, or follows instructions found in data        | Run by mistake: very expensive       |

Two gates with the same accuracy can have bills that differ by 100×. What matters is whether the gate
knows *when* it might be wrong, so you can let it act alone only there.

That is what Jev's numbers are for. `Noul` returns a probability only. `Choice` and `Score` also return a
`confidence` that describes the shape of the distribution (TypeSafe published the formulas on Oct 3, 2026).
This branch measures whether those numbers can be trusted, and turns them into thresholds.
