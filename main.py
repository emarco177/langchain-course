from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


def main():
    # 1️⃣ Carga variables de entorno (.env)
    load_dotenv()

    print("🚀 Hello from langchain-course! - LangChain Agents")

    # 2️⃣ Definir el prompt base
    information = "elon musk information"

    summary_template = """Given the information {information} about a person,
    create:
    1. A short summary.
    2. A fun fact about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # 3️⃣ Detectar si hay API Key de Groq (si no, usar Ollama)
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        print("⚡ Using Groq API (cloud model)")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # ✅ modelo Groq más reciente
            api_key=groq_api_key,
            temperature=0.3
        )
    else:
        print("🧠 Using Ollama (local model)")
        llm = ChatOllama(
            model="gemma3:1b",  # ✅ modelo que ya descargaste
            temperature=0.7
        )

    # 4️⃣ Conectar el prompt con el modelo
    chain = summary_prompt_template | llm

    # 5️⃣ Ejecutar
    response = chain.invoke({"information": information})

    print("\n🤖 Respuesta del agente:\n")
    print(response.content)


if __name__ == "__main__":
    main()
