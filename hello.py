import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
file_path = 'd:/Auto_Rec/values.csv'

df = pd.read_csv(file_path)

# Keep only relevant columns (as you did)
df = df.iloc[:, 12:]

print("Initial Shape:", df.shape)

# =========================
# 2. CLEAN DATA
# =========================

# Normalize strings (strip spaces, lower case)
df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else x)

# Replace ALL possible missing indicators
missing_values = [
    'NC', 'NSU', '', ' ', 'NA', 'N/A', 'na', 'n/a',
    'null', 'Null', 'NULL', 'nan', 'NaN',
    'never tried', 'i don’t know', "i don't know"
]

df = df.replace(missing_values, np.nan)

# Convert everything to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# =========================
# 3. DEBUG MISSING VALUES
# =========================
total_cells = df.size
nan_count = df.isnull().sum().sum()

print("\n--- Missing Value Debug ---")
print(f"Total Cells: {total_cells}")
print(f"NaN Count: {nan_count}")
print(f"NaN Percentage: {nan_count / total_cells:.4f}")

# =========================
# 4. FILTER USERS (OPTIONAL)
# =========================
max_nan_ratio = 0.4
max_nan_allowed = int(df.shape[1] * max_nan_ratio)

df = df[df.isnull().sum(axis=1) <= max_nan_allowed]
df.reset_index(drop=True, inplace=True)

print("\nAfter Filtering Shape:", df.shape)

# =========================
# 5. COMPUTE STATISTICS
# =========================

num_users = df.shape[0]
num_items = df.shape[1]

total_possible = num_users * num_items

# Stack (ignores NaNs automatically)
df_stacked = df.stack().reset_index(name='Score')

observed = len(df_stacked)

density = observed / total_possible
sparsity = 1.0 - density

# =========================
# 6. RATING STATS
# =========================

raw_avg = df_stacked['Score'].mean()

# Normalize
df_stacked['Score_norm'] = df_stacked['Score'] / 5.0
norm_avg = df_stacked['Score_norm'].mean()

# =========================
# 7. FINAL OUTPUT
# =========================

print("\n--- FINAL DATASET STATISTICS ---")
print(f"Number of Users (m): {num_users}")
print(f"Number of Items (n): {num_items}")
print(f"Total Possible Interactions: {total_possible}")
print(f"Observed Interactions: {observed}")
print(f"Density: {density:.6f}")
print(f"Sparsity: {sparsity:.6f}")
print(f"Average Rating (raw scale): {raw_avg:.4f}")
print(f"Average Rating (normalized): {norm_avg:.4f}")

# =========================
# 8. SANITY CHECK
# =========================

if density == 1.0:
    print("\n⚠️ WARNING: Density is 1 → No missing values detected!")
    print("Check if missing values were properly replaced.")

# =========================
# 9. SAVE CLEAN DATA (OPTIONAL)
# =========================
df.to_csv('cleaned_dataset.csv', index=False)