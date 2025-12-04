# show_all_columns.py
import pandas as pd

df = pd.read_csv("final_training_dataset.csv")

print(f"Total columns: {len(df.columns)}")
print("\n" + "="*80)
print("ALL COLUMN NAMES:")
print("="*80)

for i, col in enumerate(df.columns, 1):
    print(f"{i:4d}. {col}")

# Save to file
with open("all_columns.txt", "w") as f:
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:4d}. {col}\n")

print("\n✅ Saved to: all_columns.txt")