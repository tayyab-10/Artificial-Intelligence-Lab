import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing images
DATA_DIR = './data'
EXPECTED_FEATURES = 42  # Expected number of features (21 landmarks × 2 coordinates)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        # Read image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with Mediapipe Hands
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Check if data is complete before appending
            if len(data_aux) == EXPECTED_FEATURES:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Skipped image {img_path} in {dir_}: incomplete data with {len(data_aux)} features.")

# Convert labels to integers using LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

class_counts = Counter(labels)
valid_classes = [label for label, count in class_counts.items() if count > 1]

# Filtering
valid_data = []
valid_labels = []
for i in range(len(labels)):
    if labels[i] in valid_classes:
        valid_data.append(data[i])
        valid_labels.append(labels[i])

valid_data = np.array(valid_data)
valid_labels = np.array(valid_labels)

# Reshape data for CNN
valid_data_reshaped = valid_data.reshape(-1, 21, 2, 1)  # 21 landmarks, 2 coordinates, 1 channel
valid_labels_categorical = to_categorical(valid_labels)  # Convert labels to one-hot encoding

# Splitting Dataset
x_train, x_test, y_train, y_test = train_test_split(valid_data_reshaped, valid_labels_categorical, test_size=0.2, shuffle=True, stratify=valid_labels)

# Initialize traditional models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

accuracy_results = {}
unique_labels = np.unique(valid_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
valid_labels_mapped = np.array([label_mapping[label] for label in valid_labels])

# Update train-test split with new labels
x_train, x_test, y_train, y_test = train_test_split(
    valid_data_reshaped, 
    valid_labels_mapped, 
    test_size=0.2, 
    shuffle=True, 
    stratify=valid_labels_mapped
)

# One-hot encoding for CNN
y_train_categorical = to_categorical(y_train, num_classes=len(unique_labels))
y_test_categorical = to_categorical(y_test, num_classes=len(unique_labels))

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Flatten the training and testing data for traditional models
    x_train_flat = x_train.reshape(x_train.shape[0], -1)  # Flatten for traditional models
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Fit the model using the raw integer labels
    model.fit(x_train_flat, y_train)  # Use integer labels directly
    y_pred = model.predict(x_test_flat)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    accuracy_results[model_name] = {
        "accuracy": accuracy * 100,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print(f"Accuracy of {model_name}: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%\n")


# CNN Model for Gesture Recognition
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train CNN
print("Training CNN...")
cnn_model.fit(
    x_train, y_train_categorical,
    epochs=20, batch_size=32,
    validation_data=(x_test, y_test_categorical),
    callbacks=[early_stopping]
)


cnn_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Evaluate CNN
cnn_accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)[1] * 100
print(f"CNN Accuracy: {cnn_accuracy:.2f}%")

accuracy_results["CNN"] = {
    "accuracy": cnn_accuracy
}

# RNN Model for Gesture Recognition
print("Training RNN...")
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(21, 2)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rnn_model.fit(x_train.reshape(-1, 21, 2), y_train_categorical, epochs=20, batch_size=32, validation_data=(x_test.reshape(-1, 21, 2), y_test_categorical), callbacks=[early_stopping])

# Evaluate RNN
rnn_accuracy = rnn_model.evaluate(x_test.reshape(-1, 21, 2), y_test_categorical, verbose=0)[1] * 100
print(f"RNN Accuracy: {rnn_accuracy:.2f}%")

accuracy_results["RNN"] = {
    "accuracy": rnn_accuracy
}

# LSTM Model for Gesture Recognition
print("Training LSTM...")
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(21, 2)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lstm_model.fit(x_train.reshape(-1, 21, 2), y_train_categorical, epochs=20, batch_size=32, validation_data=(x_test.reshape(-1, 21, 2), y_test_categorical), callbacks=[early_stopping])

# Evaluate LSTM
lstm_accuracy = lstm_model.evaluate(x_test.reshape(-1, 21, 2), y_test_categorical, verbose=0)[1] * 100
print(f"LSTM Accuracy: {lstm_accuracy:.2f}%")

accuracy_results["LSTM"] = {
    "accuracy": lstm_accuracy
}

# Save the best model
best_model_name = max(accuracy_results, key=lambda x: accuracy_results[x]['accuracy'])
if best_model_name == "CNN":
    cnn_model.save('best_cnn_model.h5')
elif best_model_name == "RNN":
    rnn_model.save('best_rnn_model.h5')
elif best_model_name == "LSTM":
    lstm_model.save('best_lstm_model.h5')
else:
    best_model = models[best_model_name]
    with open('best_model.p', 'wb') as f:
        pickle.dump({'model': best_model}, f)

# Extract model names and their accuracies
model_names = list(accuracy_results.keys())
model_accuracies = [accuracy_results[model]['accuracy'] for model in model_names]

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(model_names, model_accuracies, marker='o', linestyle='-', color='b')

# Add graph title and labels
plt.title('Model Accuracies Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.grid(True)

# Show data points
for i, acc in enumerate(model_accuracies):
    plt.text(i, acc, f"{acc:.2f}%", ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()

# Show the plot
plt.show()
print(f"The best model is {best_model_name} with an accuracy of {accuracy_results[best_model_name]['accuracy']:.2f}%")
