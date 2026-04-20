from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Konidela Pawan Kalyan[5] (born Konidela Kalyan Babu;[8] 2 September 1971)[2] is an Indian politician, actor, philanthropist, and martial artist serving as the 11th Deputy Chief Minister of Andhra Pradesh since June 2024. He is also the Minister of Panchayat Raj, Rural Development and Rural Water Supply; Environment, Forest, Science and Technology in the Government of Andhra Pradesh as MLA representing the Pithapuram constituency.[3] He is the founding president of the Janasena Party.

As an actor, Kalyan is known for his distinctive style and mannerisms in Telugu cinema. He has been featured in Forbes India's Celebrity 100 list multiple times since 2012.[9][14] He is the recipient of a Filmfare Award and a SIIMA Award.

Kalyan made his acting debut in the 1996 film Akkada Ammayi Ikkada Abbayi. Then, he had a streak of six consecutive hits, among which Tholi Prema (1998), Thammudu (1999), Badri (2000), and Kushi (2001) became back-to-back blockbusters. In 2001, he became the first ever South Indian brand ambassador for Pepsi.[18] He made a comeback with Jalsa (2008), the highest-grossing Telugu film of that year, and continued with hits like Gabbar Singh (2012), Attarintiki Daredi (2013), Gopala Gopala (2015), and Bheemla Nayak (2022). He received the Filmfare Award for Best Actor for Gabbar Singh.

Kalyan holds a black belt in Karate.[19] He is also known for his philanthropic work, supporting various social causes.[20]

In March 2014, Pawan Kalyan founded the Janasena Party (JSP). He has been active in state politics and was elected from the Pithapuram constituency by a margin of over 70,000 votes, subsequently becoming the deputy chief minister in 2024.[21] 
"""   

    summary_template = f"""
    given the information {information} about a person, extract the following information:
    1. A short summary
    2. tell Two interesting facts about him.
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
