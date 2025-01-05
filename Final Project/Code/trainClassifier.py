import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

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

# Spliting DAtasets 
x_train, x_test, y_train, y_test = train_test_split(valid_data, valid_labels, test_size=0.2, shuffle=True, stratify=valid_labels)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

accuracy_results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy * 100
    print(f"Accuracy of {model_name}: {accuracy * 100:}%")

# Saving the best model based on the ACCUracy
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

with open('best_model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)

print(f"The best model is {best_model_name} with an accuracy of {accuracy_results[best_model_name]:.2f}%")
