import pandas as pd

results = {
    "Model": ["Baseline", "Grid Search", "Random Search"],
    "MSE": [14.3333, 14.3333, 54.4907],
    "R2 Score": [0.9709, 0.9709, 0.8894]
}

df = pd.DataFrame(results)

print("Comparative Analysis of Models")
print(df.to_string(index=False))

best_model = df.loc[df["MSE"].idxmin()]

print("\nBest Model Based on MSE:")
print(best_model)