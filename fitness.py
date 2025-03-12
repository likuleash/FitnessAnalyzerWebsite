import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your keypoints dataset (replace with actual dataset)
# Example: Assuming each sample has 17 keypoints (x, y) making 34 features per sample
# and 'labels' represent the corresponding pose labels

# Load your dataset (keypoints and corresponding pose labels)
# X -> (n_samples, n_keypoints*2), e.g., 34 keypoints (x, y) would have shape (n_samples, 34)
# y -> (n_samples,), each sample is labeled with a class (pose)
X = np.load('keypoints.npy')  # Keypoints data
y = np.load('labels.npy')      # Labels for the pose

# Step 2: Preprocess the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Build the Pose Classification Model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # Number of pose classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 6: Make Predictions (optional)
sample_keypoints = X_test[0].reshape(1, -1)  # Example of a single test sample
predicted_class = model.predict(sample_keypoints)
predicted_pose = label_encoder.inverse_transform([np.argmax(predicted_class)])
print(f"Predicted Pose: {predicted_pose[0]}")


#try 2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
#
# # Step 1: Load your keypoints dataset (replace with actual dataset)
# try:
#     X = np.load('keypoints.npy')  # Keypoints data
#     y = np.load('labels.npy')     # Labels for the pose
# except FileNotFoundError:
#     print("Dataset files not found. Please ensure 'keypoints.npy' and 'labels.npy' are in the correct directory.")
#     exit()
#
# # Step 2: Preprocess the labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#
# # Step 3: Build the Pose Classification Model
# model = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu'),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(len(np.unique(y_encoded)), activation='softmax')  # Number of pose classes
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Step 4: Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
#
# # Step 5: Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
#
# # Step 6: Make Predictions (optional)
# sample_keypoints = X_test[0].reshape(1, -1)  # Example of a single test sample
# predicted_class = model.predict(sample_keypoints)
# predicted_pose = label_encoder.inverse_transform([np.argmax(predicted_class)])
# print(f"Predicted Pose: {predicted_pose[0]}")
