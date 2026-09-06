import pandas as pd


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("dataset/spam.xlsx")


# ==========================================
# 2. SHOW ORIGINAL SHAPE
# ==========================================

print("Original dataset shape:")
print(df.shape)


# ==========================================
# 3. REMOVE MISSING VALUES
# ==========================================

df = df.dropna()


# ==========================================
# 4. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()


# ==========================================
# 5. CONVERT LABELS
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# 6. SHOW CLEAN DATA
# ==========================================

print("\nCleaned dataset:")
print(df)


# ==========================================
# 7. SHOW LABEL COUNTS
# ==========================================

print("\nLabel counts:")
print(df["label"].value_counts())


# ==========================================
# 8. SAVE CLEAN DATASET
# ==========================================

df.to_csv(
    "dataset/spam_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved!")