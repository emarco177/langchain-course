from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information_text="""
    Bison Kaalamaadan is a 2025 Indian Tamil-language sports action drama film written and directed by Mari Selvaraj. It is jointly produced by Sameer Nair, Deepak Seigal, Pa. Ranjith and Aditi Anand under Applause Entertainment and Neelam Studios. The film stars Dhruv Vikram, leading an ensemble cast featuring Pasupathy, Ameer, Lal, Anupama Parameswaran, Rajisha Vijayan and Azhagam Perumal. Based on the life of kabaddi player Manathi Ganesan, it follows a man who strives to excel in the sport while overcoming caste-based discrimination.

The film was officially announced in December 2020 under the tentative title DV03, as it is Dhruv's third film as a lead actor, and the official title was announced in May 2024. Principal photography commenced the same month in Chennai and wrapped by mid February 2025. The film has music composed by Nivas K. Prasanna, cinematography handled by Ezhil Arasu K and editing by Sakthi Thiru.

Bison Kaalamaadan was released on 17 October 2025, coinciding with Diwali. The film received positive reviews from critics and audience and became a success.[3]

Plot
Kittan Velusamy is from a socially oppressed caste in rural Tamil Nadu. Driven by a passion for kabaddi, Kittan battles caste prejudice, violent feuds, and familial resistance—particularly from his protective father, Velusamy—as he dreams of representing India at the 1994 Asian Games.

Set against the tumultuous backdrop of 1994 caste-based violence and district rivalries, Kittan's path is fraught with setbacks engineered by powerful landlords like Kandasamy, whose dominance colours every aspect of village life. The film introduces two rival leaders, Pandiaraja, a local hero, and Kandasamy, whose escalating feud affects Kittan's family and ambitions.

Kittan finds direction when a committed kabaddi coach, Kandeeban, spots his raw talent and becomes the mentor who teaches him how to turn his anger into purpose on the kabaddi court. Despite being humiliated, attacked and repeatedly reminded of his marginalised identity, Kittan refuses to bend. His resilience transforms him into a symbol of defiance and hope known as Bison. Through all of this, his sister Raaji remains the emotional centre of the household, steadying them when everything collapses around them. In the end, Kittan rises above brutal hierarchies, fear and violence to become a national kabaddi champion, representing Indian team in Asian Games.
    """

    Summary_template="""
    given the information {information_text} about a movie.  I want you to create
    1.A short summary
    2.two intresting factors about the movie
    """

    summary_prompt_template=PromptTemplate(
        input_variables=["information_text"],template=Summary_template
    )

    llm= ChatOllama(
        model="gemma3:270m",
        temperature=0,
        base_url="http://localhost:11434"
    )
    chain = summary_prompt_template | llm

    

    response = chain.invoke({"information_text": information_text})
    print(response.content)


if __name__ == "__main__":
    main()
