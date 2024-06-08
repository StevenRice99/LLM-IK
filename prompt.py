from main import generate_prompt
from configuration import ROBOTS, ORIENTATION, LIMITS

if __name__ == "__main__":
    print(generate_prompt(ROBOTS, ORIENTATION, LIMITS))
