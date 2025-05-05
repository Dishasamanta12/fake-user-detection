import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create synthetic dataset
def generate_dataset(n=1000, save_path="data/fake_user_dataset.csv"):
    np.random.seed(42)
    df = pd.DataFrame({
        'username_length': np.random.randint(5, 15, size=n),
        'has_profile_picture': np.random.randint(0, 2, size=n),
        'number_of_followers': np.random.randint(0, 1000, size=n),
        'number_of_following': np.random.randint(0, 2000, size=n),
        'posts_per_day': np.round(np.random.rand(n) * 10, 2),
        'account_age_days': np.random.randint(10, 3650, size=n),
        'is_fake': np.random.randint(0, 2, size=n)
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset generated and saved to {save_path}")

# Load dataset
def load_dataset(path="data/fake_user_dataset.csv"):
    return pd.read_csv(path)

# Train and evaluate model
def train_model(df):
    X = df.drop('is_fake', axis=1)
    y = df['is_fake']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return clf, X.columns

# Show feature importance
def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=model.feature_importances_, y=feature_names)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# Predict new user
def predict_new_user(model):
    print("\n--- Predict Fake/Real User ---")
    username_length = int(input("Enter username length: "))
    has_profile_picture = int(input("Has profile picture? (1 = Yes, 0 = No): "))
    number_of_followers = int(input("Number of followers: "))
    number_of_following = int(input("Number of following: "))
    posts_per_day = float(input("Posts per day: "))
    account_age_days = int(input("Account age (in days): "))

    new_user = [[username_length, has_profile_picture, number_of_followers,
                 number_of_following, posts_per_day, account_age_days]]
    prediction = model.predict(new_user)
    print("\nPrediction: This user is", "FAKE" if prediction[0] == 1 else "REAL")

# Main execution
if __name__ == "__main__":
    generate_dataset()  # Comment this if dataset already exists
    df = load_dataset()
    model, features = train_model(df)
    plot_feature_importance(model, features)
    predict_new_user(model)