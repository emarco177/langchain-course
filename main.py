from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print(os.environ.get('OPEN_API_KEY'))
    print('init')

if __name__ == '__main__':
    main()
