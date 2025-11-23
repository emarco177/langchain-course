import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


def main():
    print("Hello from langchain-course!")
    if not API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY in your .env file.")

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",  # pick a model from your list
        api_key=API_KEY
    )

    information="""
        Elon Reeve Musk[b] (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be around $500 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; his Canadian citizenship is congenital, his mother having been born there. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.

Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and returned to managing his companies.

Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted considerable public backlash, particularly in response to DOGE.

Early life
See also: Musk family
Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[1][2] He is of British and Pennsylvania Dutch ancestry.[3][4] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[5][6][7] Musk therefore holds both South African and Canadian citizenship from birth.[8] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[9][10][11][12]

His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Elon was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the Technocracy movement[13][14] who moved to South Africa in 1950.[15]

Elon has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[16][17][7][18] Musk was baptized as a child in the Anglican Church of Southern Africa.[19][20] Despite both Elon and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[12] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[21][22] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[1]

After his parents divorced in 1979, Elon, aged around 9, chose to live with his father because Errol Musk had an Encyclopædia Britannica and a computer.[23][3][9] Elon later regretted his decision and became estranged from his father.[24] Elon has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations.[25] In one incident, after an altercation with a fellow pupil, Elon was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries.[26] Elon described his father berating him after he was discharged from the hospital.[26] Errol denied berating Elon and claimed, "The [other] boy had just lost his father to suicide, and Elon had called him stupid. Elon had a tendency to call people stupid. How could I possibly blame that child?"[27]

Elon was an enthusiastic reader of books, and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[11][28] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[29] At age twelve, Elon sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500 (equivalent to $1,579 in 2024).[30][31]
    """

    summary_template="""
        given the information {information} about a person i want you to create :
        1 . A Short Summary
        2. Two interesting facts about them
    """
    summary_prompt_template=PromptTemplate(
        input_variables=["information"],template=summary_template
    )
    # llm =ChatOpenAI(temperature=0,model="gpt-5")


    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()







# main.py
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Load API key from .env
# load_dotenv()
# API_KEY = os.getenv("GOOGLE_API_KEY")

# if not API_KEY:
#     raise ValueError("Please set GOOGLE_API_KEY in your .env file.")

# # Initialize the LLM
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-2.5-pro",  # pick a model from your list
#     api_key=API_KEY
# )

# # Example query
# prompt = "Hello, how are you?"
# response = llm.invoke(prompt)

# # Print the response
# print("Response:", response)






# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# models = genai.list_models()
# for m in models:
#     print(m.name)
