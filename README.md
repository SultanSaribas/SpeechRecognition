# Turkish Speech Recognition

A deep-learning based speech recognition / keyword classification project for recognizing **14 Turkish command words** from audio recordings.

## Project Overview

This project was developed individually during an internship period at **GOHM Electronics & Software**. The goal was to build an end-to-end audio classification workflow for Turkish voice commands and to experiment with multiple neural-network approaches.

The repository contains implementations and experiments for:
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Transfer Learning
- Audio preprocessing and data pipeline components

## Recognized Command Words

The system is designed to classify the following 14 Turkish command words:

`Aç`, `Aşağı`, `Başlat`, `Devam`, `Dur`, `Evet`, `Geri`, `Hayır`, `İleri`, `İptal`, `Kapa`, `Sağ`, `Sol`, `Yukarı`

## Dataset

The dataset used in the project is the **Turkish Speech Command Dataset** available on Kaggle:

https://www.kaggle.com/muratkurtkaya/turkish-speech-command-dataset

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Jupyter Notebook

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. **Audio data preparation**  
   Audio samples are organized and prepared for model development.

2. **Preprocessing and feature extraction**  
   Audio data is transformed into model-ready representations using Python audio-processing tools.

3. **Model development**  
   ANN and CNN approaches are implemented and evaluated for multi-class Turkish command recognition.

4. **Transfer learning experiments**  
   Transfer-learning based approaches are explored as an alternative to models trained from scratch.

5. **Pipeline organization**  
   Reusable preprocessing and modeling steps are organized in a dedicated pipeline structure.

## Repository Structure

```text
SpeechRecognition/
├── ANN/                 # Artificial Neural Network experiments
├── CNN/                 # Convolutional Neural Network experiments
├── Pipeline/            # Data / model pipeline components
├── PythonCode/          # Supporting Python code
├── Records/             # Audio records
├── RecordsFromDataset/  # Dataset-derived recordings
├── TransferLearning/    # Transfer learning experiments
├── database/            # Project data-related files
└── README.md
```

## Running the Project

1. Clone or download the repository.
2. Download the Turkish Speech Command Dataset.
3. Place the dataset inside a folder named `dataset` in the project directory.
4. Use the relevant ANN, CNN, Pipeline, or TransferLearning notebooks/scripts for the experiment you want to run.

> Note: The repository contains multiple experimental approaches rather than a single packaged application entry point.

## My Contribution

I designed and implemented the project individually, including the machine-learning workflow for Turkish command recognition, audio preprocessing and feature extraction, ANN/CNN model experiments, transfer-learning experiments, and the organization of the project pipeline.

## Project Evidence

Full source code is publicly available at:

https://github.com/SultanSaribas/SpeechRecognition

## Author

**Sultan Sarıbaş**
