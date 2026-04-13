#%%
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Howard Phillips Lovecraft (US: /ˈlʌvkræft/; August 20, 1890 – March 15, 1937) was an American writer of weird, horror, fantasy, and science fiction. He is best known for his creation of the Cthulhu Mythos,[a] but his legacy is also apparent in terms like "Lovecraftian horror" and an enduring fandom.

    Born in Providence, Rhode Island, Lovecraft spent most of his life in New England. Following the institutionalization of his father in 1893, he lived affluently until his family's wealth dissipated after the death of his grandfather. Lovecraft then lived with his mother with reduced financial security until she too was institutionalized in 1919. He began to write essays for the United Amateur Press Association and in 1913 wrote a critical letter to a pulp magazine that ultimately led to his involvement in pulp fiction. He became active in the speculative fiction community and was published in several pulp magazines. Marrying Sonia Greene in 1924, Lovecraft moved to New York City and later became the center of a wider group of authors known as the "Lovecraft Circle". They introduced him to Weird Tales, which became his most prominent publisher. Lovecraft's time in New York took a toll on his mental state and financial conditions. He returned to Providence in 1926 and remained active as a writer for 11 years, until his death at the age of 46. It was during this final period that Lovecraft produced some of his most popular works, including The Call of Cthulhu, At the Mountains of Madness, The Shadow over Innsmouth, and The Shadow Out of Time.

    Lovecraft's literary corpus is rooted in cosmicism, which was simultaneously his personal philosophy and the main theme of his fiction. Cosmicism posits that humanity is an insignificant part of the cosmos. He incorporated fantasy and science fiction elements into his stories, representing the perceived fragility of anthropocentrism. This was tied to his ambivalent views on knowledge. His works were largely set in a fictionalized version of New England. Civilizational decline also plays a major role in his works, as he believed that the West was in decline during his lifetime. Lovecraft's early political views were conservative and traditionalist; additionally, he held a number of racist views for much of his adult life. Following the Great Depression, Lovecraft's political views became more socialist while still remaining elitist and aristocratic.

    Throughout his adult life, Lovecraft was never able to support himself from his earnings as an author and editor. He was virtually unknown during his lifetime, and was almost exclusively published in pulp magazines before his death. A scholarly revival of Lovecraft's work began in the 1970s, and he is now regarded as one of the most significant 20th-century authors of supernatural horror fiction. Many direct adaptations and spiritual successors followed. Works inspired by Lovecraft, adaptations or original works, began to form the basis of the Cthulhu Mythos, which utilizes Lovecraft's characters, setting, and themes.
    """

    summary_template = """
    You are given the following factual information about a real person.

    {information}

    Your task:
    1. Write a short factual summary of this person.
    2. Provide two interesting facts based ONLY on the information above.

    Do NOT invent or create fictional content.
    Do NOT add information not present in the text.

    Format:

    Summary:
    ...

    Interesting Facts:
    - ...
    - ...
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template= summary_template
    )

    llm = ChatOpenAI(temperature=0, model = "gpt-4o-mini")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()

# %%
