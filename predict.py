import joblib


# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================

model = joblib.load(
    "models/spam_model.pkl"
)

print("Model loaded successfully!")


# ==========================================
# 2. GET MESSAGE FROM USER
# ==========================================

message = input(
    "\nEnter an email/message: "
)


# ==========================================
# 3. PREDICT
# ==========================================

prediction = model.predict(
    [message]
)[0]


# ==========================================
# 4. GET PROBABILITY
# ==========================================

probability = model.predict_proba(
    [message]
)[0]


# ==========================================
# 5. DISPLAY RESULT
# ==========================================

if prediction == 1:

    print("\n🚨 SPAM")

    print(
        "Spam probability:",
        round(probability[1] * 100, 2),
        "%"
    )

else:

    print("\n✅ NOT SPAM")

    print(
        "Not Spam probability:",
        round(probability[0] * 100, 2),
        "%"
    )