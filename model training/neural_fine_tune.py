import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.optimizers import Adam

# Load the previously trained model and preprocessors
model = load_model('models/my_model_st.keras')
scaler = joblib.load('models/scaler_st.pkl')
encoder = joblib.load('models/encoder_st.pkl')

# Optionally load a different dataset for fine-tuning
file_path_fine_tune = 'AD_dataset/st_del.csv'
fine_tune_dataset = pd.read_csv(file_path_fine_tune)

#fine_tune_dataset=fine_tune_dataset.fillna(0)

X_fine_tune = fine_tune_dataset.drop(['Label','Flags'], axis=1)
y_fine_tune = fine_tune_dataset['Label']

# Split the fine-tuning dataset
X_train_ft, X_test_ft, y_train_ft, y_test_ft = train_test_split(X_fine_tune, y_fine_tune, test_size=0.20, random_state=42)

# Normalize and encode the fine-tuning data using the loaded scaler and encoder
X_train_ft_scaled = scaler.transform(X_train_ft)
X_test_ft_scaled = scaler.transform(X_test_ft)
y_train_encoded_ft = encoder.transform(y_train_ft.values.reshape(-1, 1)).toarray()
y_test_encoded_ft = encoder.transform(y_test_ft.values.reshape(-1, 1)).toarray()

# Lower the learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001),  # Reduced learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

# Fine-tuning the model
model.fit(X_train_ft_scaled, y_train_encoded_ft, validation_split=0.2, epochs=50, batch_size=32)  # Fewer epochs and possibly smaller batch size

# Evaluate the model after fine-tuning
evaluation_ft = model.evaluate(X_test_ft_scaled, y_test_encoded_ft)
#print(f'After Fine-Tuning - Test Loss: {evaluation_ft[0]}, Test Accuracy: {evaluation_ft[1]}, Precision: {evaluation_ft[2]}, Recall: {evaluation_ft[3]}')
print(f'After Fine-Tuning - Test Loss: {evaluation_ft[0]}, Test Accuracy: {evaluation_ft[1]}')

# Save the fine-tuned model
model.save('models/my_model_fine_tuned_st.keras')
