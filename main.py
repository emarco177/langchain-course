from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def create_llm(llm = 'ollama_model', ollama_model='qwen2.5:7b'):
    if llm == 'ollama_model':
        return ChatOllama(
            model=ollama_model,
            base_url='http://local-ia-ubuntu:11434',
            temperature=0,
        ) 

    return ChatOpenAI(
        model='gpt-4.1-mini',
        temperature=0,
    )
   

def main():
    information = """
        Elon Reeve Musk FRS (Pretória, 28 de junho de 1971) é um empresário,[4] empreendedor, 
        inventor e filantropo sul-africano, naturalizado canadense e estadunidense. 
        É o fundador, diretor executivo e diretor técnico da SpaceX; 
        CEO da Tesla, Inc.; um dos cofundadores da OpenAI, fundador e CEO da Neuralink; cofundador, 
        presidente da SolarCity, proprietário do X (antigo Twitter) e ex-comissário do Departamento 
        de Eficiência Governamental dos Estados Unidos durante o segundo governo Trump em 2025. 
        Desde 2021, Musk se posiciona como a pessoa mais rica do mundo, até então com um patrimônio 
        líquido estimado em 200 bilhões de dólares, de acordo com o Bloomberg Billionaires Index. 
        Já a revista Forbes estimou sua fortuna em 221,3 bilhões de dólares, principalmente de suas 
        participações acionárias nas empresas Tesla e na SpaceX.[5] Em junho de 2026, a Forbes estimou 
        que seu patrimônio líquido seja de US$ 1.4 trilhão, sendo o primeiro trilionário em termos nominais
        absolutos, além de ser o único registrado oficialmente na história moderna.
    """
    summary_prompt_template = """
        Dada a informação abaixo sobre uma pessoa:
        {information}

        Crie, em português do Brasil:
        1. Um resumo curto
        2. Dois fatos interessantes sobre ela
        
        O formato de saida deve ser um json, chave do #1. 'resumo' e chave do #2 'fatos'.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=['information'], template=summary_prompt_template
    )
    
    llm = create_llm()

    chain = summary_prompt_template | llm

    # response = chain.invoke(input={'information': information})
    for chunk in chain.stream({'information': information}):
        print(chunk.content, end='', flush=True)
    print()

if __name__ == '__main__':
    main()
