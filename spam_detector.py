# =====================================================
#              SPAM MESSAGE DETECTOR
# Developed By: Kashish Nihalani
# Internship Task - SoftGrowTech
# =====================================================

import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class SpamDetector:

    def __init__(self, file_name):

        self.file_name = file_name
        self.model = None
        self.accuracy = 0

    # ----------------------------------
    # Read Dataset
    # ----------------------------------

    def load_dataset(self):

        try:

            data = pd.read_csv(self.file_name)

            if "Label" not in data.columns or "Message" not in data.columns:
                raise Exception("Dataset format is incorrect.")

            return data

        except FileNotFoundError:

            print("\nDataset not found.")
            exit()

    # ----------------------------------
    # Clean Text
    # ----------------------------------

    def clean_text(self, text):

        text = str(text).lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ----------------------------------
    # Train Model
    # ----------------------------------

    def train_model(self):

        print("\nTraining model... Please wait.\n")

        data = self.load_dataset()

        data["Message"] = data["Message"].apply(self.clean_text)

        X = data["Message"]

        y = data["Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        self.model = Pipeline([
            (
                "vectorizer",
                TfidfVectorizer(stop_words="english")
            ),
            (
                "classifier",
                MultinomialNB()
            )
        ])

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        self.accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("Model trained successfully.")

        print(
            f"\nAccuracy : {self.accuracy*100:.2f}%"
        )

        print("\nClassification Report\n")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        print("\nConfusion Matrix\n")

        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        joblib.dump(
            self.model,
            "spam_model.pkl"
        )

        print("\nModel saved successfully.")

    # ----------------------------------
    # Load Saved Model
    # ----------------------------------

    def load_model(self):

        try:

            self.model = joblib.load(
                "spam_model.pkl"
            )

            print("\nSaved model loaded successfully.")

        except:

            print("\nNo trained model found.")
            print("Please train the model first.")

    # ----------------------------------
    # Predict Single Message
    # ----------------------------------

    def predict_message(self):

        if self.model is None:

            print("\nModel not loaded.")
            return

        message = input("\nEnter Message : ")

        cleaned = self.clean_text(message)

        prediction = self.model.predict(
            [cleaned]
        )[0]

        probability = self.model.predict_proba(
            [cleaned]
        )[0]
        
        
        spam_probability = probability[list(self.model.classes_).index("spam")] * 100
        ham_probability = probability[list(self.model.classes_).index("ham")] * 100

        print("\n" + "=" * 45)

        if prediction == "spam":
            print("Prediction        : SPAM")
        else:
            print("Prediction        : NOT SPAM")

        print(f"Spam Probability : {spam_probability:.2f}%")
        print(f"Ham Probability  : {ham_probability:.2f}%")
        print("=" * 45)

    # ----------------------------------
    # Batch Prediction
    # ----------------------------------

    def batch_prediction(self):

        if self.model is None:
            print("\nModel not loaded.")
            return

        try:
            total = int(input("\nHow many messages do you want to check? : "))

            for i in range(total):

                print(f"\nMessage {i+1}")

                msg = input("Enter Message : ")

                cleaned = self.clean_text(msg)

                result = self.model.predict([cleaned])[0]

                if result == "spam":
                    print("Result : SPAM")
                else:
                    print("Result : NOT SPAM")

                print("-" * 35)

        except ValueError:

            print("Please enter a valid number.")

    # ----------------------------------
    # Show Accuracy
    # ----------------------------------

    def show_accuracy(self):

        if self.accuracy == 0:
            print("\nAccuracy not available.")
            print("Please train the model first.")
        else:
            print(f"\nCurrent Accuracy : {self.accuracy*100:.2f}%")



# ==========================================
# Main Program
# ==========================================

def main():

    detector = SpamDetector("spam.csv")

    while True:

        print("\n")
        print("=" * 45)
        print("        SPAM MESSAGE DETECTOR")
        print("=" * 45)
        print("1. Train Model")
        print("2. Load Saved Model")
        print("3. Check Single Message")
        print("4. Check Multiple Messages")
        print("5. View Accuracy")
        print("6. Exit")
        print("=" * 45)

        choice = input("Enter Choice : ")

        if choice == "1":

            detector.train_model()

        elif choice == "2":

            detector.load_model()

        elif choice == "3":

            detector.predict_message()

        elif choice == "4":

            detector.batch_prediction()

        elif choice == "5":

            detector.show_accuracy()

        elif choice == "6":

            print("\nThank you for using Spam Message Detector.")
            break

        else:

            print("\nInvalid Choice. Please try again.")


if __name__ == "__main__":
    main()