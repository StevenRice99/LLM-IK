from main import load_model, view

if __name__ == "__main__":
    # Pass the name of the folder under "Models" for the robot you want.
    model, data, lower, upper, site, path, solvers = load_model("1DOF")
    if model is not None:
        view(model, data)
