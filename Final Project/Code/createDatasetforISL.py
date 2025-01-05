import pandas as pd
import pickle

csv_file_path = './Indian_Sign_Language_Dataset.csv'
df = pd.read_csv(csv_file_path)

# Extract features and labels
data = df.iloc[:, :-1].values  # All columns except the last (features)
labels = df.iloc[:, -1].values  # The last column (labels)

# Save the dataset as a pickle file
with open('indian_sign_language_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Indian Sign Language dataset saved. Total samples: {len(data)}")
