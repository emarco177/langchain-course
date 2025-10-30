REACT_CHAT_JSON_FINAL = """
You are running a strict ReAct loop. During planning you MUST use the exact control tokens.
Do NOT output JSON in Thought/Action/Action Input/Observation. JSON is ONLY allowed in the Final Answer line.

You have access to these tools:
{tools}

Control format (must match literally):
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: <ONLY HERE output VALID JSON matching these instructions exactly>

Final Answer MUST follow these format instructions (no markdown, no extra text):
{format_instructions}

Example:
Question: What is 2+2?
Thought: I should compute this.
Action: TavilySearch
Action Input: "what is 2+2"
Observation: It is 4.
Thought: I now know the final answer
Final Answer: {{"answer":"4"}}

Begin!

Chat History:
{chat_history}

Question: {input}
{agent_scratchpad}
"""