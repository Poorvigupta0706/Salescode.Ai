# Salescode.Ai

 # Spot the Fake Photo

A lightweight computer vision project that classifies an image as either a **real photo** or a **photo of a screen (recapture)**.

## Features

- FFT (Frequency Analysis)
- LBP (Texture Features)
- Glare Detection
- Canny Edge Density
- SVM Classifier


<img width="1536" height="1024" alt="ChatGPT Image Jul 1, 2026, 10_38_39 AM" src="https://github.com/user-attachments/assets/43c20751-5173-4af7-a3ab-8819c29a1aba" />

## Project Structure

dataset/
├── train/
│   ├── real/
│   └── fake/

features.py
train.py
predict.py
```

## Why SVM?

SVM was chosen because it performs well on small datasets and works effectively with handcrafted features such as FFT, LBP, glare detection, and edge density. It provides fast inference, is lightweight, and outputs a probability score between **0** (real photo) and **1** (photo of a screen), making it well-suited for this task.

## Train

```bash
python train.py
```
## Predict

```bash
python predict.py image.jpg
```
Output:
- `0` → Real Photo
- `1` → Photo of a Screen

#Here are my model acuuracies

<img width="1920" height="1080" alt="Screenshot (354)" src="https://github.com/user-attachments/assets/017237a5-bad2-4226-9692-d961f45540b4" />

<img width="1920" height="1080" alt="Screenshot (353)" src="https://github.com/user-attachments/assets/b8155e04-4a34-4a23-bf87-40177a39f251" />

## Tech Stack

- Python
- OpenCV
- NumPy
- scikit-image
- scikit-learn
