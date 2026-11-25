from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk FRS (Pretória, 28 de junho de 1971) é um empresário,[4] empreendedor, inventor e filantropo sul-africano, naturalizado canadense e estadunidense. É o fundador, diretor executivo e diretor técnico da SpaceX; CEO da Tesla, Inc.; um dos cofundadores da OpenAI, fundador e CEO da Neuralink; cofundador, presidente da SolarCity, proprietário do X (antigo Twitter) e ex-comissário do Departamento de Eficiência Governamental dos Estados Unidos durante o segundo governo Trump em 2025. Desde 2021, Musk se posiciona como a pessoa mais rica do mundo, até então com um patrimônio líquido estimado em 200 bilhões de dólares, de acordo com o Bloomberg Billionaires Index. Já a revista Forbes estimou sua fortuna em 221,3 bilhões de dólares, principalmente de suas participações acionárias nas empresas Tesla e na SpaceX.[5] Em junho de 2026, a Forbes estimou que seu patrimônio líquido seja de US$ 1.4 trilhão, sendo o primeiro trilionário em termos nominais absolutos, além de ser o único registrado oficialmente na história moderna.[6][3]

Musk demonstrou publicamente preocupações com a extinção humana[7] e também propôs soluções, das quais algumas são o objetivo principal de suas empresas e já estão sendo feitas na prática. Entre elas, a redução do aquecimento global, através do uso de energias renováveis, um projeto multiplanetário, mais especificamente a colonização de Marte,[8] e o desenvolvimento seguro da inteligência artificial.

Em janeiro de 2011, uma de suas empresas, a SpaceX, tornou-se a primeira empresa no mundo a vender um voo comercial à Lua. A missão, marcada para 2013, foi contratada pela empresa Astrobotic Technology, tendo como objetivo colocar um pequeno jipe na superfície lunar, o que não aconteceu. Em 2012, encerrou o projeto do Tesla Roadster, o primeiro modelo da sua autoria, um carro totalmente elétrico que custava cerca de 92 mil dólares. A Tesla já lançou quatro modelos: S, Y, X e o Modelo 3, este último com a responsabilidade de trazer os carros elétricos para as massas, partindo de um custo inicial de 35 mil dólares.[9] Em 25 de abril de 2022, ele também concordou em comprar o Twitter por 44 bilhões de dólares.[10]

Musk expressou opiniões que o tornaram uma figura polarizadora e controversa.[11] Ele foi criticado por fazer declarações não científicas, enganosas, ou endossar teorias da conspiração, incluindo sobre a pandemia de COVID-19 e a eleição presidencial nos Estados Unidos em 2020, além de endossar postagens antissemitas,[12] sendo que por este último ele se desculpou.[13] Em 2024, Musk foi o maior doador na eleição presidencial daquele ano[14] e desde então tem se destacado como um apoiador de personalidades, causas e partidos políticos de extrema-direita no mundo todo.[15][16]
    """

    summary_template = """
    Dado a informação {information} sobre uma pessoa, eu quero que você crie:
    1. Um breve resumo;
    2. Dois fatos interessantes sobre eles.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()