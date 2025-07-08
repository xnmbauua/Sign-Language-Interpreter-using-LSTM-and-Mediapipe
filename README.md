# Sign Language Interpreter (Real-time Gesture Recognition)

This project implements a **real-time sign language interpreter** using **LSTM (Long Short-Term Memory)** neural networks and **MediaPipe** for dynamic hand gesture recognition. It also includes a few Bhutanese gestures in Dzongkha language as part of the training set to support local customization.

## Features

- Real-time gesture recognition using webcam
- Hand landmark detection powered by **MediaPipe**
- Temporal gesture learning using **LSTM**
- Modular code structure with custom dataset collection and training pipeline

## Project Files

- `collection.py` – Captures gesture data using webcam and saves it as dataset
- `process.py` – Preprocesses the collected data into sequences for training
- `train.py` – Trains the LSTM model on processed sequences
- `real.py` – Runs real-time gesture recognition using the trained model and webcam

## Tech Stack

- Python
- OpenCV
- NumPy
- MediaPipe
- TensorFlow / Keras (for LSTM model)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sign-language-interpreter.git
   cd sign-language-interpreter
