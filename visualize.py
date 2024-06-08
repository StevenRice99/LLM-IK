from main import load_model, view
from configuration import ROBOTS, ORIENTATION, LIMITS, COLLISIONS

if __name__ == "__main__":
    model, data, lower, upper, site, path, solvers = load_model(ROBOTS, ORIENTATION, LIMITS, COLLISIONS)
    if model is not None:
        view(model, data)
