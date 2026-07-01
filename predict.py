import os
import joblib
import numpy as np

from features import extract_features

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


# -------------------------
# PREDICT ONE IMAGE
# -------------------------
def predict_image(image_path):

    features = extract_features(image_path)
    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 0:
        label = "REAL"
    else:
        label = "SCREEN"

    return label, probability


# -------------------------
# TEST COMPLETE FOLDER
# -------------------------
def test_folder(folder_path):

    total = 0
    correct = 0

    # Automatically detect expected label
    folder_name = os.path.basename(folder_path).lower()

    if folder_name == "real":
        expected = "REAL"
    elif folder_name == "screen":
        expected = "SCREEN"
    else:
        expected = None

    print("\nTesting Folder:", folder_path)
    print("-" * 60)

    for file in os.listdir(folder_path):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, file)

        try:

            label, prob = predict_image(image_path)

            print(f"{file}")
            print(f"Prediction : {label}")
            print(f"Real   : {prob[0]*100:.2f}%")
            print(f"Screen : {prob[1]*100:.2f}%")
            print()

            total += 1

            if expected is not None and label == expected:
                correct += 1

        except Exception as e:
            print(f"{file} -> ERROR: {e}")

    print("-" * 60)
    print("Total Images :", total)

    if expected is not None:
        print("Correct      :", correct)
        print("Accuracy     : {:.2f}%".format(correct * 100 / total))


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    dataset_folder = input(
        "Enter folder path (real or screen): "
    ).strip()

    test_folder(dataset_folder)