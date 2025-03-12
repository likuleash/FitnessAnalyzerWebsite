import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 1. Load Dataset
file_path = "fitness_motion_dataset.csv"
try:
    data = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Data Preprocessing
X = data.iloc[:, :-1].values  # Features (all columns except the last)
y = data.iloc[:, -1].values   # Labels (last column)

# Normalize feature values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels to numeric format
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Reshape data for LSTM input
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 3. Model Development Using TensorFlow Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(25),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Number of classes in y
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Train Model
history = model.fit(X_train_reshaped, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# 5. Evaluate Model (TensorFlow Model)
y_pred_tf = np.argmax(model.predict(X_test_reshaped), axis=1)
accuracy_tf = accuracy_score(y_test, y_pred_tf)
print(f"\nTensorFlow Model Accuracy: {accuracy_tf:.2f}")

# Confusion Matrix for TensorFlow Model
conf_matrix_tf = confusion_matrix(y_test, y_pred_tf)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_tf, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("TensorFlow Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Training Progress Visualization (TensorFlow Model)
# Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('TensorFlow Model - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Training and Validation Loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('TensorFlow Model - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 7. Testing Phase for Other Models
models = {
    'Support Vector Machine': SVC(),
    'Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB()
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} - Testing Phase Accuracy: {accuracy:.2f}")

    # Display Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 8. Final Results Comparison
print("\nFinal Model Accuracy Comparison:")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.2f}")
print(f"TensorFlow Model: {accuracy_tf:.2f}")
