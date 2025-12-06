import pandas as pd
from GA import select
import matplotlib.pyplot as plt
import numpy as np

"""----------------------------------------------------------------------
 Part 1: Real Data (Baseball)
----------------------------------------------------------------------"""
df = pd.read_csv("data/baseball.dat", sep=r"\s+")

y = df["salary"].values
X_df = df.drop(columns=["salary"])
X = X_df.values
feature_names = X_df.columns

print("\n--- Baseball Data Example ---")
print("Dataset shape:", df.shape)

result = select(X, y,
                n_pop=40,
                n_gen=200,
                mutation_rate=0.05,
                crossover_rate=0.8,
                lambda_penalty=0.05,
                cv=5,
                random_state=11,
                feature_names=feature_names)

# Plot real data fitness trace
plt.figure(figsize=(9, 5))
plt.plot(result["fitness_history"], marker="o", alpha=0.6)
plt.xlabel("Generation")
plt.ylabel("Best CV R²")
plt.title("GA Fitness Over Generations (Baseball Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_history_baseball.png", dpi=150)
plt.show()

print("Saved: fitness_history_baseball.png")

print("Number of selected variables:", result["n_selected"])
print("Selected variable names:", result["selected_var_names"])
print("Best CV R²:", round(result["best_fitness"], 4))


"""----------------------------------------------------------------------
 Part 2: Simulated Data
----------------------------------------------------------------------"""
print("\n--- Simulated Data Example ---")

rng = np.random.RandomState(123)
n_samples, n_features = 300, 27
X_sim = rng.randn(n_samples, n_features)
y_sim = 2.0 * X_sim[:, 0] - 1.0 * X_sim[:, 2] + 0.1 * rng.randn(n_samples)

feature_names_sim = [f"X{i}" for i in range(n_features)]

result_sim = select(X_sim, y_sim,
                    n_pop=30,
                    n_gen=40,
                    mutation_rate=0.05,
                    crossover_rate=0.8,
                    lambda_penalty=0.05,
                    cv=3,
                    random_state=42,
                    feature_names=feature_names_sim)

# Plot simulated data fitness trace
plt.figure(figsize=(9, 5))
plt.plot(result_sim["fitness_history"], marker="o", alpha=0.6, color="darkgreen")
plt.xlabel("Generation")
plt.ylabel("Best CV R²")
plt.title("GA Fitness Over Generations (Simulated Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_history_simulated.png", dpi=150)
plt.show()

print("Saved: fitness_history_simulated.png")

print("Selected Vars:", result_sim["selected_var_names"])
print("Num Selected:", result_sim["n_selected"])
print("Best CV R²:", round(result_sim["best_fitness"], 4))

























# import pandas as pd
# from GA import select
# import matplotlib.pyplot as plt
# import numpy as np


# """Here we test the GA on the baseball data"""
# "----------------------------------------------------------------------"
# # Load data
# df = pd.read_csv("data/baseball.dat", sep=r"\s+")

# # Separate predictors and response
# y = df["salary"].values
# X_df = df.drop(columns=["salary"])  # Only drop once

# X = X_df.values
# feature_names = X_df.columns  # Reuse the same dropped dataframe

# result = select(X, y,
#                 n_pop=40,
#                 n_gen=200,
#                 mutation_rate=0.05,
#                 crossover_rate=0.8,
#                 lambda_penalty=0.0,
#                 cv=5,
#                 random_state=11,
#                 feature_names=feature_names)

# plt.figure(figsize=(8, 5))
# plt.plot(result["fitness_history"], marker="o")
# plt.xlabel("Generation")
# plt.ylabel("Best CV R²")
# plt.title("GA Fitness Over Generations")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("fitness_history.png", dpi=150)
# plt.show()

# print("Saved fitness plot to fitness_history.png")

# print("\nNumber of selected variables:", result["n_selected"])
# print("Selected variable names:", result["selected_var_names"])
# print("Best CV R²:", result["best_fitness"])


# print("\n--- Simulated Data Example ---")
# "----------------------------------------------------------------------"

# # Synthetic dataset where only X0 and X2 matter
# rng = np.random.RandomState(123)
# n_samples, n_features = 300, 27
# X_sim = rng.randn(n_samples, n_features)
# y_sim = 2.0 * X_sim[:, 0] - 1.0 * X_sim[:, 2] + 0.1 * rng.randn(n_samples)

# feature_names_sim = [f"X{i}" for i in range(n_features)]

# result_sim = select(X_sim, y_sim,
#                     n_pop=30,
#                     n_gen=30,
#                     mutation_rate=0.05,
#                     crossover_rate=0.8,
#                     cv=3,
#                     random_state=42,
#                     feature_names=feature_names_sim)

# print("Selected Vars:", result_sim["selected_var_names"])
# print("Num Selected:", result_sim["n_selected"])
# print("Best CV R²:", round(result_sim["best_fitness"], 4))
