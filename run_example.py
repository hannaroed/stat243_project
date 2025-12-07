import argparse
import pandas as pd
from GA import select
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# HOW TO RUN THIS SCRIPT (from terminal):
#
# Default model (Linear Regression):
#     python run_example.py
#
# Choose a different prediction model:
#
#     python run_example.py --model ridge
#     python run_example.py --model lasso
#     python run_example.py --model elasticnet
#     python run_example.py --model rf
#
# Valid options for --model :
#     linear   (default)
#     ridge
#     lasso
#     elasticnet
#     rf       (Random Forest)
#
# Example with Ridge Regression:
#     python run_example.py -m ridge
#
# NOTES:
# - Random Forest ("rf") is typically MUCH slower than the linear
#   models, especially with larger populations / more generations.
#
# The chosen model will be used inside the GA fitness calculation,
# and the plot titles + output filenames will reflect the model used.
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GA variable selection on baseball and simulated data "
                    "with selectable prediction model."
    )
    parser.add_argument(
        "-m", "--model",
        choices=["linear", "ridge", "lasso", "elasticnet", "rf"],
        default="linear",
        help="Prediction model to use inside GA fitness "
             "(default: linear)."
    )
    return parser.parse_args()


def pretty_model_name(model_type: str) -> str:
    mapping = {
        "linear": "Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "elasticnet": "Elastic Net",
        "rf": "Random Forest"
    }
    return mapping.get(model_type, model_type)


def main():
    args = parse_args()
    model_type = args.model
    model_label = pretty_model_name(model_type)

    print(f"Using model: {model_label} ({model_type})")

    """------------------------------------------------------------------
     Part 1: Real Data (Baseball)
    ------------------------------------------------------------------"""
    df = pd.read_csv("data/baseball.dat", sep=r"\s+")

    y = df["salary"].values
    X_df = df.drop(columns=["salary"])
    X = X_df.values
    feature_names = X_df.columns

    print("\n--- Baseball Data Example ---")
    print("Dataset shape:", df.shape)

    result = select(
        X, y,
        n_pop=40,
        n_gen=200,
        mutation_rate=0.05,
        crossover_rate=0.8,
        lambda_penalty=0.05,
        cv=5,
        random_state=11,
        feature_names=feature_names,
        model_type=model_type,
        # optionally: model_kwargs={"alpha": 1.0} for ridge/lasso, etc.
        model_kwargs=None
    )

    # Plot real data fitness trace
    plt.figure(figsize=(9, 5))
    plt.plot(result["fitness_history"], marker="o", alpha=0.6)
    plt.xlabel("Generation")
    plt.ylabel("Best penalized CV R²")
    plt.title(f"GA Fitness Over Generations (Baseball, {model_label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fitness_history_baseball_{model_type}.png", dpi=150)
    plt.show()

    print(f"Saved: fitness_history_baseball_{model_type}.png")

    print("Number of selected variables:", result["n_selected"])
    print("Selected variable names:", result["selected_var_names"])
    print("Best penalized CV R²:", round(result["best_fitness"], 4))

    """------------------------------------------------------------------
     Part 2: Simulated Data
    ------------------------------------------------------------------"""
    print("\n--- Simulated Data Example ---")

    rng = np.random.RandomState(123)
    n_samples, n_features = 300, 27
    X_sim = rng.randn(n_samples, n_features)
    y_sim = 2.0 * X_sim[:, 0] - 1.0 * X_sim[:, 2] + 0.1 * rng.randn(n_samples)

    feature_names_sim = [f"X{i}" for i in range(n_features)]

    result_sim = select(
        X_sim, y_sim,
        n_pop=30,
        n_gen=40,
        mutation_rate=0.05,
        crossover_rate=0.8,
        lambda_penalty=0.05,
        cv=3,
        random_state=42,
        feature_names=feature_names_sim,
        model_type=model_type,
        model_kwargs=None
    )

    # Plot simulated data fitness trace
    plt.figure(figsize=(9, 5))
    plt.plot(result_sim["fitness_history"], marker="o", alpha=0.6)
    plt.xlabel("Generation")
    plt.ylabel("Best penalized CV R²")
    plt.title(f"GA Fitness Over Generations (Simulated, {model_label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fitness_history_simulated_{model_type}.png", dpi=150)
    plt.show()

    print(f"Saved: fitness_history_simulated_{model_type}.png")

    print("Selected Vars:", result_sim["selected_var_names"])
    print("Num Selected:", result_sim["n_selected"])
    print("Best penalized CV R²:", round(result_sim["best_fitness"], 4))


if __name__ == "__main__":
    main()
