import pandas as pd
from sklearn.model_selection import train_test_split
import joblib  # For saving StandardScaler and OneHotEncoder
from tensorflow.keras.models import load_model  # For saving and loading the Keras model


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Part 2: Evaluate Model on New Dataset

# Load the model and preprocessors
model = load_model('models/my_model_incremental_st.keras')
scaler = joblib.load('models/scaler_st.pkl')
encoder = joblib.load('models/encoder_st.pkl')

# Load new dataset
new_dataset_path = 'AD_dataset/st_repadshift.csv'
new_dataset = pd.read_csv(new_dataset_path)

#new_dataset = new_dataset.fillna(0)

# Preprocess the new dataset
X_new = new_dataset.drop(['Label','Flags'], axis=1)
y_new = new_dataset['Label']
X_new_scaled = scaler.transform(X_new)
y_new_encoded = encoder.transform(y_new.values.reshape(-1, 1)).toarray()

# Evaluate the model on the new dataset
evaluation_new = model.evaluate(X_new_scaled, y_new_encoded)
print()
print(f'New Test Loss: {evaluation_new[0]}, New Test Accuracy: {evaluation_new[1]}, Precision: {evaluation_new[2]}, Recall: {evaluation_new[3]}')


# Predict on the new dataset for confusion matrix and classification report
y_pred_new = model.predict(X_new_scaled)
y_pred_classes_new = np.argmax(y_pred_new, axis=1)
y_true_classes_new = np.argmax(y_new_encoded, axis=1)

conf_matrix_new = confusion_matrix(y_true_classes_new, y_pred_classes_new)

target_classes = [str(label) for label in encoder.categories_[0]]
report_new = classification_report(y_true_classes_new, y_pred_classes_new, target_names=target_classes)

print("\nNew Test Confusion Matrix:")
print(conf_matrix_new)
print("\nNew Test Classification Report:")
print(report_new)




