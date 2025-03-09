import os
import sys
import argparse
import json

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Load configuration from config.json
def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the full workflow: Training, Testing, Prediction, and Evaluation.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file.")
    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)
    print("Configuration loaded successfully.")

    # Run training
    print("\nStarting Training...")
    os.system(f"python train.py")  
    print("Training Completed!\n")

    # Run testing
    print("\nStarting Testing...")
    os.system(f"python test.py")
    print("Testing Completed!\n")

    # Run prediction
    print("\nRunning Prediction...")
    os.system(f"python prediction.py")
    print("Prediction Completed!\n")

    # Run evaluation
    print("\nRunning Evaluation...")
    os.system(f"python evaluation.py")
    print("Evaluation Completed!\n")

if __name__ == "__main__":
    main()
