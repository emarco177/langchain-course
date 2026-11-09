from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

load_dotenv()


def main() -> None:
    print("Hello from langchain-project!")

    information = """
Democrats made significant gains in the 2026 midterms, raising questions about how the results could affect AI regulation, infrastructure investment and the technology industry’s political strategy. An anxious market reacted with extreme volatility; the Nasdaq went up by 1.4% on the day after the election, only to come down by 5.5% the day after. But the AI trade is different from the long-term prospects of AI as a transformative technology. For this, the change in the political environment ushered in by the blue wave may be good news.
The Risks Of The AI Boom
The AI infrastructure buildout has been the engine of the United States’ GDP growth this year. Stijn Van Nieuwerburgh, a professor at Columbia Business School, writing for the Brookings Papers on Economic Activity, estimates that financing the AI buildout will require $10.3 trillion from 2025 to 2032, about 3.6% of US GDP per year. Larry Fink, CEO of BlackRock, sounded the alarm on the Wall Street Journal editorial pages: “When all that investment demand collides with a limited supply of capital, borrowing becomes more expensive for everyone.”


The financial news cycle revolves around the AI industry and its supply chain. Data center demand seems infinite, capped by suppliers’ ability to produce components, a lack of skilled labor to stand up major compute plants and an aging energy grid struggling with the needs of a country accelerating on AI while trying to recreate its lost manufacturing capacity. The administration saw in AI an opportunity to demonstrate America’s superiority in a geopolitical race against China while driving the economy. The technology industry, a direct beneficiary, jumped to support the messaging.


The most accelerationist camps of the tech sector, notably a group of venture capitalists with a direct interest in a light regulatory environment and a race towards investment returns, secured positions of power in the government and drove the agenda in the first eighteen months of the second Trump administration. Other technology executives publicly aligned themselves with the administration’s AI agenda.

Frontier AI labs, now some of the largest corporations in the history of American capitalism, continued to behave like immature startups. OpenAI and Anthropic each disclosed that their models breached outside organizations during testing. OpenAI called its Hugging Face incident accidental, and Anthropic traced its four incidents to a misconfigured test environment compounded by model misjudgment. The press quickly attributed the incidents to rogue AI, a technology almost sentient.



At the same time, company leaders talked about building an alien mind and threatened the risk of human extinction. Oh, but humanity could still be saved, if only you would continue to fund growth so that we could be the ones creating a superintelligence.

At the center of the action, Nvidia, the most valuable company in the world, reached a market valuation of more than five trillion in a neck-and-neck race with Germany for being the third “largest economy.” The company acts as the supplier and the bank for the AI boom, with a web of stakes, loans and guarantees across the AI industry.

The Billie: Get A Look Inside The Minds Of Billionaires

 


The Growing Backlash Against Data Centers
Many feared that the political pushback against AI would come from existential risk concerns. I don’t. People rarely vote on abstract issues. But especially in the last year, AI became a kitchen table topic. Families ask if their energy bills are going to be affected or if their communities’ zoning plans will change to accommodate a new data center. They are concerned about their jobs and whether AI will affect their kids’ ability to get a job after graduating from college. They are discussing the environmental impact of local projects. These are issues that mobilize the electorate and will increase the pressure on policymakers to act.

However, my take is that these practical concerns don’t fully explain why we saw dozens of data center-related measures on the ballot in the recent election. I believe data centers have also become a focal point for broader frustration with the current administration, the technology industry’s political influence and the impact of AI infrastructure on local communities.

The tech industry made a political mistake, and AI is paying the price. The price tag would have been much lower if, instead of advocating for light regulation, the industry had engaged in earlier stakeholder management. In this context, OpenAI’s laudable effort with a “new standard for responsible AI infrastructure,” published two days after the election, risks sounding opportunistic and late.


Why Regulation Could Benefit AI
AI can have an incredible positive economic impact. It can increase productivity, transform the nature of work, help address previously intractable problems, democratize access to specialized services and even transform governments’ ability to deliver services. To realize this promise, the current AI bubble cannot burst. It needs, instead, to adjust away from business models that do not close and create an environment of trust in which both consumer and enterprise demand continue to grow. Real, sustainable and profitable demand is the true engine for the infrastructure buildout that is powering the economy.

A stable, predictable and properly regulated environment enables growth. The approach to AI policy that the blue wave seems to usher in can push us in that direction. This is not a time for extreme politics or measures. Neither unfettered laissez-faire nor unrealistic proposals to ban the technology are what the country needs.

There are bipartisan federal proposals such as the FRONTIER Act that can be built upon to address the vacuum of national policy. In California, home of the largest companies in the field, a group of experts convened by the governor will deliver recommendations this month on amending California’s AI safety laws, including independent verification. The election results could create new momentum for AI regulation, although whether lawmakers can agree on a framework—and what it would mean for the industry—remains uncertain.
    """

    summary_template = """
        given the information {information} about a topic I want you to create:
        1. A short summary
        2. two interesting facts about the topic
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )


    llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=1000)

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.text)
    
if __name__ == "__main__":
    main()
