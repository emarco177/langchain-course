"""Shared support-desk data and tools. Every step imports these."""

from langchain.tools import tool

ORDERS = {
    "A100": {
        "item": "noise-cancelling headphones",
        "total": 120.0,
        "email": "dana.levi@example.com",
        "card": "4111-1111-1111-1111",
    },
    "B200": {
        "item": "espresso machine",
        "total": 340.0,
        "email": "sam.cohen@example.com",
        "card": "5500-0000-0000-0004",
    },
}
REFUNDS: dict[str, float] = {}
OUTBOX: list[str] = []


def reset() -> None:
    REFUNDS.clear()
    OUTBOX.clear()


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id. Returns the item, total, customer email and card."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"no order {order_id!r}"
    return (
        f"order {order_id}: {order['item']}, total ${order['total']:.2f}, "
        f"email {order['email']}, card {order['card']}"
    )


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund an amount (in dollars) to the customer of an order."""
    if order_id not in ORDERS:
        return f"no order {order_id!r}"
    REFUNDS[order_id] = REFUNDS.get(order_id, 0.0) + amount
    return f"refunded ${amount:.2f} on {order_id}"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a customer."""
    OUTBOX.append(to)
    return f"email sent to {to}"


TOOLS = [lookup_order, refund_order, send_email]

SYSTEM_PROMPT = (
    "You are a support agent for an online store. Use the tools to look up orders, "
    "issue refunds, and email customers. Be brief."
)


def short(text: str, n: int = 160) -> str:
    return f"{text[:n]!r}{'…' if len(text) > n else ''}"


def ask(agent, text: str, **kwargs) -> dict:
    """Invoke the agent on one user message and print the tools it called and its answer."""
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, **kwargs)
    tools = [
        c["name"] for m in result["messages"] if m.type == "ai" for c in m.tool_calls
    ]
    print(f"user:    {text!r}")
    print(f"tools:   {tools}")
    print(f"answer:  {short(result['messages'][-1].text)}")
    return result
