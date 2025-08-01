import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the original dataset
                # original_dataset
                # ori_dset_cutime
original_dataset_path = "AD_dataset/ori_dset_cutime.csv"  
original_dataset = pd.read_csv(original_dataset_path)

# Preprocess the original dataset
X_orig = original_dataset.drop(['Label'], axis=1)
y_orig = original_dataset['Label']

# Split the original dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.20, stratify=y_orig, random_state=42)

# Normalize the training data
scaler_orig = StandardScaler()
X_train_scaled = scaler_orig.fit_transform(X_train)
X_test_scaled = scaler_orig.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=53, random_state=30)
rf_model.fit(X_train_scaled, y_train)

# Print the feature importances
feature_importances = rf_model.feature_importances_
print("Feature importances:")
for i, feature in enumerate(X_train.columns):
    print(f"{feature}: {feature_importances[i] * 100:.2f}%")
print()


# Evaluate the model on the original dataset's test set
y_pred_test = rf_model.predict(X_test_scaled)
print(f'Accuracy on the original dataset test set: {accuracy_score(y_test, y_pred_test)}')
print()
print(classification_report(y_test, y_pred_test))

# Load the new dataset for testing
new_dataset_path = 'AD_dataset/delay.csv'  
new_dataset = pd.read_csv(new_dataset_path)

# Preprocess the new dataset (using the scaler fitted on the original dataset)
X_new = new_dataset.drop(['Label'], axis=1)
y_new = new_dataset['Label']
X_new_scaled = scaler_orig.transform(X_new)

# Test the model on the new dataset
y_new_pred = rf_model.predict(X_new_scaled)

# Evaluate the model's performance on the new dataset
print(f'Accuracy on the new dataset: {accuracy_score(y_new, y_new_pred)}')
print()
print(classification_report(y_new, y_new_pred))
