# Emotion Recognition from Speech using Deep Learning

This repository contains our solution for a hands-on assignment in Deep Learning, focused on building a custom convolutional neural network to predict emotional valence from raw audio signals.

Objective
Develop a deep learning model that can estimate emotional valence (scale 1-5) from spoken utterances using end-to-end audio processing without a pre-trained model

Project Highlights
Custom Model Design: Built and trained a neural network from scratch using PyTorch, capable of handling variable-length audio input.

Audio Preprocessing: Implemented feature extraction and normalization techniques to prepare speech data for model input.

Model Architecture: Explored multiple architectures such as CNNs to capture temporal and spectral patterns in speech.

Evaluation: Model tested on a blind test set and compared against a provided baseline for leaderboard scoring.

Performance Metric: Emotional valence prediction evaluated using standard regression metrics; Dice or Jaccard used for segmentation metrics when applicable.

Dataset
Format: Audio utterances (44.1 kHz) provided as .pkl files containing NumPy arrays and corresponding valence scores.

Challenge: Input variability in both length and speaker; goal is to build a generalizable model for unseen speech.

Technologies Used
Python, PyTorch
NumPy, SciPy, Librosa
Matplotlib, Seaborn 
