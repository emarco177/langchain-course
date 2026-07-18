import os

from dotenv import load_dotenv

# Improt .env variables
load_dotenv()


def main():
    print("Hello from lanchain-couse")
    print(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
