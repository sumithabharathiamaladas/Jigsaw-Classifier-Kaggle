import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import os

# --- 1. Load Data ---
# Note: This assumes the CSV files are in the same folder as this script.
# WARNING: Replace the paths below with the ABSOLUTE paths you copied!
TRAIN_PATH = r"C:\Users\sumitha.amaladas\Downloads\jigsaw_classifier\train.csv"
TEST_PATH = r"C:\Users\sumitha.amaladas\Downloads\jigsaw_classifier\test.csv"

try:
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    print("✅ Data files loaded successfully using absolute paths.")
# ...
except FileNotFoundError:
    print("❌ Error: train.csv or test.csv not found.")
    print(f"Current working directory: {os.getcwd()}")
    print("Please ensure the files are in the same directory as this script.")
    exit()

# Define features and target
X_train_body = df_train['body'].fillna('')
X_test_body = df_test['body'].fillna('')
X_train_rule = df_train[['rule']]
X_test_rule = df_test[['rule']]
y_train = df_train['rule_violation']

# --- 2. Feature Engineering ---

# 2a. TF-IDF on 'body' (comment text)
print("⚙️ Fitting TF-IDF on comment body...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_body_tfidf = tfidf.fit_transform(X_train_body)
X_test_body_tfidf = tfidf.transform(X_test_body)

# 2b. One-hot encode 'rule'
print("⚙️ One-Hot Encoding 'rule'...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_rule_ohe = encoder.fit_transform(X_train_rule)
X_test_rule_ohe = encoder.transform(X_test_rule)

# 2c. Combine features
print("⚙️ Combining features...")
X_train_final = hstack([X_train_body_tfidf, X_train_rule_ohe])
X_test_final = hstack([X_test_body_tfidf, X_test_rule_ohe])
print(f"Final training feature shape: {X_train_final.shape}")

# --- 3. Model Training ---

# Use Logistic Regression as a baseline classifier
print("⚙️ Training Logistic Regression model...")
model = LogisticRegression(solver='liblinear', random_state=42, C=1.0)
model.fit(X_train_final, y_train)

# --- 4. Prediction ---

# Predict the probability of the positive class (rule_violation = 1)
print("⚙️ Generating predictions on test data...")
test_predictions_proba = model.predict_proba(X_test_final)[:, 1]

# --- 5. Create Submission File ---

submission_df = pd.DataFrame({
    'row_id': df_test['row_id'],
    'rule_violation': test_predictions_proba
})

submission_file_name = "submission.csv"
submission_df.to_csv(submission_file_name, index=False)

print("\n--- Process Complete ---")
print(f"✅ Submission file saved as: {submission_file_name}")