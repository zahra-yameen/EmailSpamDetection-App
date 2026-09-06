import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("dataset/spam.xlsx")

print("Dataset loaded successfully!")


# ==========================================
# 2. DISPLAY FIRST 5 ROWS
# ==========================================

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 3. DATASET SHAPE
# ==========================================

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 4. COLUMN NAMES
# ==========================================

print("\nColumn names:")
print(df.columns)


# ==========================================
# 5. DATA INFORMATION
# ==========================================

print("\nDataset information:")
df.info()


# ==========================================
# 6. MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 7. DUPLICATES
# ==========================================

print("\nDuplicates before removing:")
print(df.duplicated().sum())


# ==========================================
# 8. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()


print("\nDuplicates after removing:")
print(df.duplicated().sum())


# ==========================================
# 9. LABEL DISTRIBUTION
# ==========================================

print("\nSpam / Not Spam distribution:")
print(df["label"].value_counts())


# ==========================================
# 10. LABEL PERCENTAGE
# ==========================================

print("\nClass percentage:")
print(
    df["label"].value_counts(normalize=True) * 100
)


# ==========================================
# 11. GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    x="label",
    data=df
)

plt.title("Spam vs Not Spam")

plt.xlabel("Email Type")

plt.ylabel("Number of Messages")

plt.show()