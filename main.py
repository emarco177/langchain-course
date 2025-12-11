import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """Bobi Wine (real name Robert Kyagulanyi Ssentamu) is a prominent Ugandan activist, pop musician, and politician who is a leading critic and opponent of the country's long-serving president, Yoweri Museveni. He is the leader of the National Unity Platform political party and the People Power movement. 
     Career Overview
  Musician & Activist: Known for his socially conscious music, which he calls "edutainment," Wine's songs focus on the struggles of underprivileged Ugandans and call for social change. His music became a form of peaceful protest against corruption and the lack of human rights in Uganda.
   Politician: Wine served as a Member of Parliament for the Kyadondo East constituency from 2017 to 2021. In 2019, he announced his candidacy for the 2021 presidential election, where he ran against President Museveni. He and his supporters heavily disputed the official election results.
Opposition Leader: 
As of December 2025, Bobi Wine continues his political campaigning across Uganda for the upcoming 2026 elections, despite ongoing government crackdowns and security force interference. Recent reports include: 
Campaign incidents: In early December 2025, Wine reported being attacked by police and security forces during campaign rallies, resulting in injuries to himself and his supporters.
Ongoing activism: His social media channels show continued engagement with supporters and a persistent call for an end to corruption and the current regime. 
"""
    summary_template = """
    Given the following information: {information},
    1. write a short story about it.
    2. Give two interesting facts about the person.
    """
    
    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"])

    ollama_llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | ollama_llm
    # openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # chain = summary_prompt_template | openai_llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
