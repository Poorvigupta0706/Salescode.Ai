import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from features import extract_features
from sklearn.model_selection import cross_val_score

def load_dataset(data_dir=r"C:\Users\dell\PycharmProjects\Salescode.Ai\ASSIGNMENT"):

    X = []
    y = []

    real_dir = os.path.join(data_dir, "real")
    screen_dir = os.path.join(data_dir, "screen")

    if not os.path.exists(real_dir):
        raise Exception(f"Real folder not found: {real_dir}")

    if not os.path.exists(screen_dir):
        raise Exception(f"Screen folder not found: {screen_dir}")

    print("REAL   :", real_dir)
    print("SCREEN :", screen_dir)

    def process_folder(folder, label):

        print("\nScanning:", folder)

        files = os.listdir(folder)

        print("Total files:", len(files))

        loaded = 0

        for file in files:

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder, file)

            try:

                features = extract_features(image_path)

                X.append(features)
                y.append(label)

                loaded += 1

                print("OK :", file)

            except Exception as e:

                print("FAILED :", file)
                print(e)

        print("Loaded:", loaded)

    # Real = 0
    process_folder(real_dir, 0)

    # Screen = 1
    process_folder(screen_dir, 1)

    print("\n--------------------------------")
    print("TOTAL IMAGES :", len(X))
    print("--------------------------------")

    return np.array(X, dtype=np.float32), np.array(y)


def train():

    print("Loading dataset...\n")

    X, y = load_dataset()

    if len(X) == 0:
        raise Exception("Dataset is empty.")

    print("\nDataset Shape:", X.shape)
    print("Labels:", np.unique(y))

    if len(np.unique(y)) < 2:
        raise Exception(
            "Need at least TWO classes (real and screen) to train an SVM."
        )

    # Create scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create model
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )

    # -----------------------------
    # 5-Fold Cross Validation
    # -----------------------------
    print("\nRunning 5-Fold Cross Validation...\n")

    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5
    )

    print("Cross Validation Scores:", scores)
    print("Average Accuracy: {:.2f}%".format(scores.mean() * 100))

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    print("\nTraining SVM...\n")
    model.fit(X_train, y_train)

    print("Training Completed.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy : {:.2f}%".format(accuracy * 100))
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\nModel Saved Successfully.")


if __name__ == "__main__":
    train()