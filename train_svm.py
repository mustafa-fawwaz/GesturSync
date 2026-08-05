import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib


def load_real_dataset(filepath="hand_gestures.csv"):
    """
    Loads the real dataset of 21 3D hand landmarks (63 features + 1 label).
    """
    print(f"Loading real dataset from {filepath}...")
    try:
        # Load the CSV without a header row
        data = pd.read_csv(filepath, header=None)
        
        # X gets all rows, and all columns EXCEPT the last one (the 63 coordinates)
        X = data.iloc[:, :-1].values 
        
        # y gets all rows, but ONLY the last column (the 0-4 gesture labels)
        y = data.iloc[:, -1].values  
        
        return X, y
    except FileNotFoundError:
        print(f"Error: {filepath} not found! Make sure you ran data_collector.py.")
        exit()


def train_and_evaluate():
    # 1. Load Data
    # 1. Load Data
    X, y = load_real_dataset("hand_gestures.csv")

    # 2. Train-Test Split (80-20 split as defined in CP_Submission-3)
    print("Performing 80-20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Model Initialization (Linear SVM with C=1.0)
    print("Initializing Support Vector Machine (Linear Kernel, C=1.0)...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)

    # 4. Training
    print("Training model...")
    svm_model.fit(X_train, y_train)

    # 5. Prediction & Evaluation
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print("\n" + "=" * 40)
    print("MODEL EVALUATION METRICS")
    print("=" * 40)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Swipe L', 'Swipe R', 'Pinch Open', 'Pinch Close', 'Neutral']))

    # 6. Save the trained model
    joblib.dump(svm_model, 'gestursync_svm_model.pkl')
    print("Model saved to 'gestursync_svm_model.pkl'")


if __name__ == "__main__":
    train_and_evaluate()
