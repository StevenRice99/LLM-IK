from main import generate_prompt


if __name__ == "__main__":
    # Pass the name of the folder under "Models" for the robot you want.
    print(generate_prompt("2DOF", False, False))
