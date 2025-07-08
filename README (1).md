
# 🤖 Hand Gesture Recognition Using Computer Vision for Sign Language

A deep learning and computer vision-based system that interprets hand gestures in real-time to facilitate communication between sign language users and non-signers. This B.Tech final-year project focuses on building an accessible and accurate sign language translator using MediaPipe and LSTM.


## 📌 Overview

Millions of individuals around the world rely on sign language, yet communication barriers persist. Our system bridges this gap by translating American Sign Language (ASL) gestures into text or speech using a webcam, MediaPipe Holistic, and a deep learning model based on LSTM.

---

## 🧠 Technologies Used

- **Languages**: Python
- **Libraries/Frameworks**: OpenCV, TensorFlow, NumPy, MediaPipe, PyTorch, Matplotlib
- **ML Models**: LSTM, CNN
- **Supporting Tools**: Anaconda, GitHub, Jupyter Notebook, pyttsx3 (TTS)

---

## 📁 Dataset

### 📚 Public Datasets
- ASL Alphabet (Kaggle)
- RWTH-BOSTON-104 / PHOENIX-2014T
- Sign Language MNIST

### 🎥 Custom Dataset
- Collected using webcam and OpenCV
- Each gesture: 30 videos × 30 frames
- Stored as `.npy` arrays of 126 features per frame

---

## 🏗️ Model Architecture

```
Input: Sequence of 30 frames
↓
MediaPipe Holistic → Extract 3D hand landmarks
↓
Preprocessing → Normalize, Augment, Flatten
↓
LSTM Layers (64, 128 units)
↓
Dense Layers (64 → 32)
↓
Output Layer (Softmax - 10 classes)
↓
Text Output + Text-to-Speech (TTS)
```

---

## 🎯 System Requirements

| Component       | Minimum Specification         |
|----------------|-------------------------------|
| CPU            | Intel Core i5 or higher        |
| RAM            | 8 GB                           |
| Storage        | 256 GB SSD                     |
| Camera         | HD Webcam                      |
| OS             | Windows 10/11                  |
| Tools          | Python 3.7+, VSCode/Jupyter    |

---

## 📊 Evaluation Metrics

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 94.6%  |
| Precision    | 94.3%  |
| Recall       | 94.2%  |
| F1-Score     | 94.1%  |
| Balanced Accuracy | ~94% |

Low confusion matrix values confirm robustness, even across similar gestures.

---

## 📈 Comparative Performance

| Approach        | Accuracy | Technique                      |
|-----------------|----------|-------------------------------|
| HMM (Zhao et al., 2015)         | 85%      | Gesture-based (HMM)             |
| CNN + HMM (Wang & Lee, 2017)    | 89%      | Hybrid                         |
| CNN on RGB (Sharma, 2020)       | 92%      | Vision-based                   |
| Sensor Glove + SVM (Zhou, 2019)| 95%      | Wearable-based                 |
| **Our Model (MediaPipe + LSTM)**| **94.6%**| Webcam-based, real-time       |

---

## 🎬 Real-Time Pipeline

1. Capture live video
2. Extract 21 × 2 × 3 = 126 keypoint features (x, y, z)
3. Feed into trained LSTM model
4. Display text + play audio (pyttsx3)

---

## 🌍 Future Enhancements

- Add support for other sign languages (BSL, ISL, etc.)
- Integrate facial expression recognition
- Optimize model size for mobile/edge deployment
- Multilingual translation
- Open-source community contributions

---

## 📦 Requirements

To run this project locally, ensure you have the following dependencies installed:

```bash
python==3.7+
numpy
opencv-python
mediapipe
tensorflow
torch
pyttsx3
matplotlib
```
You can install all requirements using the following command:

```bash
pip install -r requirements.txt
```

---
