from main import load_model, view
from configuration import ROBOT

if __name__ == "__main__":
    model, data, lower, upper, site, path, solvers = load_model(ROBOT)
    if model is not None:
        view(model, data)
