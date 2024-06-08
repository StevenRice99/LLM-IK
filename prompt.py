from main import generate_prompt
from configuration import ROBOT, ORIENTATION, LIMITS

if __name__ == "__main__":
    print(generate_prompt(ROBOT, ORIENTATION, LIMITS))
