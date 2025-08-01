# Part 1: Train and Save Model & Preprocessors

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib  # For saving StandardScaler and OneHotEncoder
from tensorflow.keras.models import load_model  # For saving and loading the Keras model
import tensorflow as tf


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

            # original_dataset
            # ori_dset_cutime
# Load and prepare the initial dataset
file_path = 'AD_dataset/ad_ori.csv'
dataset = pd.read_csv(file_path)

#dataset = dataset.fillna(0)

X = dataset.drop(['Label','Flags'], axis=1)
y = dataset['Label']

# Split the dataset first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Normalize the features with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode the labels with OneHotEncoder
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1)).toarray()

# Define and compile the model
num_classes = y_train_encoded.shape[1]
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
     Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer=Adam(), 
    loss='categorical_crossentropy',
    metrics=['accuracy', 
    tf.keras.metrics.Precision(), 
    tf.keras.metrics.Recall()])

# Train the model
model.fit(X_train_scaled, y_train_encoded, validation_split=0.2, epochs=330, batch_size=32)

# Evaluate and print initial accuracy
evaluation = model.evaluate(X_test_scaled, y_test_encoded)
print()
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


"""
# Save the model and preprocessors
model.save('models/my_model_del.keras')
joblib.dump(scaler,'models/scaler_del.pkl')
joblib.dump(encoder,'models/encoder_del.pkl')

"""