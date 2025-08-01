import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

# Load the model and preprocessors
model = load_model('models/my_model_st.keras')
scaler = joblib.load('models/scaler_st.pkl')
encoder = joblib.load('models/encoder_st.pkl')

# Load new or additional dataset for incremental training
file_path_new_data = 'AD_dataset/st_pad.csv'
new_dataset = pd.read_csv(file_path_new_data)

#new_dataset=new_dataset.fillna(0)

X_new = new_dataset.drop(['Label','Flags'], axis=1)
y_new = new_dataset['Label']

# Split the new dataset first
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.20, random_state=42)

# Normalize and encode the new training and testing data using the loaded scaler and encoder
X_train_new_scaled = scaler.transform(X_train_new)
X_test_new_scaled = scaler.transform(X_test_new)
y_train_encoded_new = encoder.transform(y_train_new.values.reshape(-1, 1)).toarray()
y_test_encoded_new = encoder.transform(y_test_new.values.reshape(-1, 1)).toarray()

# Incremental Training
model.fit(X_train_new_scaled, y_train_encoded_new, validation_split=0.2, epochs=50, batch_size=32)

# Evaluate the model after incremental training
evaluation_new = model.evaluate(X_test_new_scaled, y_test_encoded_new)
print(f'After Incremental Training - Test Loss: {evaluation_new[0]}, Test Accuracy: {evaluation_new[1]}, Precision: {evaluation_new[2]}, Recall: {evaluation_new[3]}')

# Save the incrementally trained model if needed
model.save('models/my_model_incremental_st.keras')
