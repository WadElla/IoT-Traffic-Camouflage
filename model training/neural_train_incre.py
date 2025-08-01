# Part 1: Train and Save Model & Preprocessors

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import joblib  # For saving StandardScaler and OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf

# Load and prepare the initial dataset
file_path = 'dataset_split/original_train.csv'
dataset = pd.read_csv(file_path)
X = dataset.drop(['Label', 'Destination Port', 'total_packets'], axis=1)
y = dataset['Label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Normalize the features with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode the labels with OneHotEncoder
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1)).toarray()

# Define and compile the model with L2 regularization and dropout
num_classes = y_train_encoded.shape[1]
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer=Adam(), 
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall()]
)

# Train the model
model.fit(X_train_scaled, y_train_encoded, validation_split=0.2, epochs=100, batch_size=32)

# Evaluate and print initial accuracy
evaluation = model.evaluate(X_test_scaled, y_test_encoded)
print(f'Initial Test Loss: {evaluation[0]}, Initial Test Accuracy: {evaluation[1]}, Precision: {evaluation[2]}, Recall: {evaluation[3]}')

# Predict on the test set to calculate confusion matrix and classification report
y_pred_test = model.predict(X_test_scaled)
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_true_classes_test = np.argmax(y_test_encoded, axis=1)

conf_matrix_test = confusion_matrix(y_true_classes_test, y_pred_classes_test)
target_classes = [str(label) for label in encoder.categories_[0]]
report_test = classification_report(y_true_classes_test, y_pred_classes_test, target_names=target_classes)

print("\nInitial Test Confusion Matrix:")
print(conf_matrix_test)
print("\nInitial Test Classification Report:")
print(report_test)

# Save the model and preprocessors
model.save('models/my_model.keras')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

# Part 2: Incremental Training with Rehearsal

# Load the model and preprocessors
from tensorflow.keras.models import load_model

model = load_model('models/my_model.keras')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

# Load new or additional dataset for incremental training
file_path_new_data = 'dataset_split/padded_train.csv'
new_dataset = pd.read_csv(file_path_new_data)
X_new = new_dataset.drop(['Label', 'Destination Port','total_packets'], axis=1)
y_new = new_dataset['Label']

# Split the new dataset
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.20, random_state=42)

# Normalize and encode the new training and testing data using the loaded scaler and encoder
X_train_new_scaled = scaler.transform(X_train_new)
X_test_new_scaled = scaler.transform(X_test_new)
y_train_encoded_new = encoder.transform(y_train_new.values.reshape(-1, 1)).toarray()
y_test_encoded_new = encoder.transform(y_test_new.values.reshape(-1, 1)).toarray()

# Combine a sample of the old data with the new training data (rehearsal)
rehearsal_sample_size = 120  # Adjust the sample size according to your needs
indices_old = np.random.choice(len(X_train_scaled), rehearsal_sample_size, replace=False)
X_rehearsal = np.concatenate((X_train_scaled[indices_old], X_train_new_scaled))
y_rehearsal = np.concatenate((y_train_encoded[indices_old], y_train_encoded_new))

# Incremental Training with combined data
model.fit(X_rehearsal, y_rehearsal, validation_split=0.2, epochs=100, batch_size=32)

# Evaluate the model after incremental training
evaluation_new = model.evaluate(X_test_new_scaled, y_test_encoded_new)
print(f'After Incremental Training - Test Loss: {evaluation_new[0]}, Test Accuracy: {evaluation_new[1]}, Precision: {evaluation_new[2]}, Recall: {evaluation_new[3]}')

# Predict on the new test set to calculate confusion matrix and classification report
y_pred_new = model.predict(X_test_new_scaled)
y_pred_classes_new = np.argmax(y_pred_new, axis=1)
y_true_classes_new = np.argmax(y_test_encoded_new, axis=1)

conf_matrix_new = confusion_matrix(y_true_classes_new, y_pred_classes_new)
report_new = classification_report(y_true_classes_new, y_pred_classes_new, target_names=target_classes)

print("\nNew Test Confusion Matrix:")
print(conf_matrix_new)
print("\nNew Test Classification Report:")
print(report_new)

# Save the incrementally trained model
model.save('models/my_model_incremental.keras')
