import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("dataset/spam_cleaned.csv")

print("Cleaned dataset loaded!")


# ==========================================
# 2. CREATE X AND y
# ==========================================

X = df["message"]

y = df["label"]


# ==========================================
# 3. CHECK DATA
# ==========================================

print("\nX:")
print(X)

print("\ny:")
print(y)


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=42,

    stratify=y
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ==========================================
# 5. CREATE ML PIPELINE
# ==========================================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=1000
        )
    )

])


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# 8. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


# ==========================================
# 9. PRECISION
# ==========================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nPrecision:")
print(precision)


# ==========================================
# 10. RECALL
# ==========================================

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nRecall:")
print(recall)


# ==========================================
# 11. F1 SCORE
# ==========================================

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nF1 Score:")
print(f1)


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(

        y_test,

        y_pred,

        labels=[0, 1],

        target_names=[
            "Not Spam",
            "Spam"
        ],

        zero_division=0
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[0, 1]
)

print("\nConfusion Matrix:")

print(cm)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(

    model,

    "models/spam_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Location: models/spam_model.pkl"
)