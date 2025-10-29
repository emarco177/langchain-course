from dotenv import load_dotenv
load_dotenv()

def get_text_length(text:str)->int:
    """ Returns the length of a text by characters"""
    return len(text)

def main():
    print("Hello from ReAct LangChain!")
    print(get_text_length(text="Dog"))

if __name__ == "__main__":
    main()
