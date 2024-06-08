from main import test_ik
from configuration import ROBOTS, ERROR, ORIENTATION, LIMITS, COLLISIONS, TESTS

if __name__ == "__main__":
    test_ik(ROBOTS, ERROR, ORIENTATION, LIMITS, COLLISIONS, True, TESTS)
