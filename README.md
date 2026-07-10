# Spam Message Detector

## Overview

The **Spam Message Detector** is a Python-based Machine Learning project that classifies SMS messages as **Spam** or **Ham** using the **TF-IDF** technique and the **Multinomial Naive Bayes** algorithm.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib

## Features

* Detects Spam and Ham messages
* Text preprocessing and cleaning
* TF-IDF text vectorization
* Machine Learning-based prediction
* Displays model accuracy
* Classification report and confusion matrix
* Supports single and multiple message prediction

## How to Run

1. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```
2. Place the `spam.csv` dataset in the project folder.
3. Run the project:

   ```
   python spam_detector.py
   ```
4. Select an option from the menu to train the model or classify messages.

## Project Structure

```text
Spam-Message-Detector/
│── spam_detector.py
│── spam.csv
│── requirements.txt
│── README.md
│── spam_model.pkl (generated automatically)
```
