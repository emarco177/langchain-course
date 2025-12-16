import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

def main():
    print("Hello from langchain-course!")
    # print("Environment variable EXAMPLE_VAR:", os.getenv("EXAMPLE_VAR"))

    info = """Google LLC is an American multinational technology corporation focused on information technology, online advertising, search engine technology, email, cloud computing, software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).[9] It has been referred to as "the most powerful company in the world" by BBC,[10] and is one of the world's most valuable brands.[11][12][13] Google's parent company Alphabet Inc. has been described as a Big Tech company.

            Google was founded on September 4, 1998, by American computer scientists Larry Page and Sergey Brin. Together, they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet.[14]

            After the success of its original service, Google Search (often known simply as "Google"), the company has rapidly grown to offer a multitude of products and services. These products address a wide range of use cases, including email (Gmail), navigation and mapping (Waze, Maps, and Earth), cloud computing (Cloud), web navigation (Chrome), video sharing (YouTube), productivity (Workspace), operating systems (Android and ChromeOS), cloud storage (Drive), language translation (Translate), photo storage (Photos), videotelephony (Meet), smart home (Nest), smartphones (Pixel), wearable technology (Pixel Watch and Fitbit), music streaming (YouTube Music), video on demand (YouTube TV), AI (Google Assistant and Gemini), machine learning APIs (TensorFlow), AI chips (TPU), and more. Many of these products and services are dominant in their respective industries, as is Google Search. Discontinued Google products include gaming (Stadia),[15] Glass, Google+, Reader, Play Music, Nexus, Hangouts, and Inbox by Gmail.[16][17] Google's other ventures outside of internet services and consumer electronics include quantum computing (Willow), self-driving cars (Waymo), and transformer models (Google DeepMind).[18]

            Google Search and YouTube are the two most-visited websites worldwide, followed by Facebook, Instagram, and ChatGPT. Google is the largest provider of search engines, mapping and navigation applications, email services, office suites, online video platforms, photo and cloud storage, mobile operating systems, web browsers, machine learning frameworks, and AI virtual assistants in the world as measured by market share.[19] On the list of the most valuable brands, Google is ranked second by Forbes as of January 2022,[20] and is fourth by Interbrand as of February 2022.[21] The company has received criticism involving issues such as privacy concerns, tax avoidance, censorship, search neutrality, antitrust, and abuse of its monopoly position.[22]"""

    summary_template ="""
    Given the information about {company}, provide a concise summary in 2 points highlighting its key aspects.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["company"],
        template=summary_template
    )   
    
    llm = ChatOllama(model="gemma3:270m", temperature=0) # doesnt give the correct output - trade off when using open weights model - smaller models are faster but less accurate
    # llm = ChatGroq(
    #     model="llama-3.1-8b-instant",  #gives accurate output
    #     temperature=0
    # )
    chain = summary_prompt_template | llm # use lcel(langchain expression language) operator to chain prompt template and llm together. The pipe operator '|' is used to create a new runnable chain by connecting the output of one component to the input of another

    response = chain.invoke(input={"company": info})
    print(response.content)
    
if __name__ == "__main__":
    main()