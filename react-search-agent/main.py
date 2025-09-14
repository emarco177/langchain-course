from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI 
from langchain_tavily import TavilySearch

tools = [
    TavilySearch()
]

model = ChatOpenAI(model="gpt-4",temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent=create_react_agent(llm=model,tools=tools,prompt=react_prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

chain = agent_executor



def main():
    result=chain.invoke({"input":"search for 3 job postings for data scientist in bay area on linkedin and summarize them in a table with columns for job title, company name, location, and job description"})
    print(result)


if __name__ == "__main__":
    main()
