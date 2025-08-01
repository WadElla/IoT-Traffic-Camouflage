import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import tensorflow as tf

# Load and prepare the original dataset
file_path = 'AD_dataset/original_dataset.csv'
dataset = pd.read_csv(file_path)

# Prepare features and labels for the original dataset
X = dataset.drop(['Label'], axis=1)
y = dataset['Label']

# Load and prepare the obfuscated dataset
obfuscated_file_path = 'AD_dataset/delay.csv' 
obfuscated_dataset = pd.read_csv(obfuscated_file_path)
X_obfuscated = obfuscated_dataset.drop(['Label'], axis=1)
y_obfuscated = obfuscated_dataset['Label']

# Normalize the features with StandardScaler (use the same scaler for both datasets)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_obfuscated_scaled = scaler.transform(X_obfuscated)

# Encode the labels with OneHotEncoder
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
y_obfuscated_encoded = encoder.transform(y_obfuscated.values.reshape(-1, 1)).toarray()

# Define stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results list to store metrics for both datasets
results = []

# Function to calculate confidence intervals with adjustment for zero standard deviation
def confidence_interval(scores):
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    
    # If standard deviation is zero, set it to 1e-6 to avoid zero std
    if std_dev == 0:
        std_dev = 1e-6
    
    if len(scores) > 1:
        ci = stats.t.interval(0.95, len(scores) - 1, loc=mean_score, scale=std_dev / np.sqrt(len(scores)))
    else:
        ci = (mean_score, mean_score)  # In case of only one value, CI can't be calculated
    
    return mean_score, std_dev, ci

# Initialize lists to store results for original and obfuscated datasets
accuracy_scores_orig = []
precision_scores_orig = []
recall_scores_orig = []
f1_scores_orig = []

accuracy_scores_obf = []
precision_scores_obf = []
recall_scores_obf = []
f1_scores_obf = []

# Loop through each fold in the StratifiedKFold
for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y)):
    print(f"************** Fold {fold + 1} **************")
    
    # Split data into training and testing for this fold (original dataset)
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]
    
    # Define and compile the neural network model
    num_classes = y_train_fold.shape[1]
    model = Sequential([
        Input(shape=(X_train_fold.shape[1],)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    # Train the model on the original dataset's training fold
    model.fit(X_train_fold, y_train_fold, validation_split=0.03, epochs=330, batch_size=32, verbose=0)
    
    # Predict and evaluate on the original test fold
    y_pred_fold = model.predict(X_test_fold)
    y_pred_classes_fold = np.argmax(y_pred_fold, axis=1)
    y_true_classes_fold = np.argmax(y_test_fold, axis=1)
    
    # Calculate metrics for the original dataset test fold
    accuracy_orig = accuracy_score(y_true_classes_fold, y_pred_classes_fold)
    precision_orig = precision_score(y_true_classes_fold, y_pred_classes_fold, average='macro')
    recall_orig = recall_score(y_true_classes_fold, y_pred_classes_fold, average='macro')
    f1_orig = f1_score(y_true_classes_fold, y_pred_classes_fold, average='macro')
    
    # Store metrics for the original dataset (across all folds)
    accuracy_scores_orig.append(accuracy_orig)
    precision_scores_orig.append(precision_orig)
    recall_scores_orig.append(recall_orig)
    f1_scores_orig.append(f1_orig)
    
    # Test the trained model on the obfuscated dataset
    y_pred_obfuscated = model.predict(X_obfuscated_scaled)
    y_pred_classes_obf = np.argmax(y_pred_obfuscated, axis=1)
    y_true_classes_obf = np.argmax(y_obfuscated_encoded, axis=1)
    
    # Calculate metrics for the obfuscated dataset
    accuracy_obf = accuracy_score(y_true_classes_obf, y_pred_classes_obf)
    precision_obf = precision_score(y_true_classes_obf, y_pred_classes_obf, average='macro')
    recall_obf = recall_score(y_true_classes_obf, y_pred_classes_obf, average='macro')
    f1_obf = f1_score(y_true_classes_obf, y_pred_classes_obf, average='macro')
    
    # Store metrics for the obfuscated dataset
    accuracy_scores_obf.append(accuracy_obf)
    precision_scores_obf.append(precision_obf)
    recall_scores_obf.append(recall_obf)
    f1_scores_obf.append(f1_obf)
    
    print(f"Original Dataset - Accuracy: {accuracy_orig}, Precision: {precision_orig}, Recall: {recall_orig}, F1-Score: {f1_orig}")
    print(f"Obfuscated Dataset - Accuracy: {accuracy_obf}, Precision: {precision_obf}, Recall: {recall_obf}, F1-Score: {f1_obf}")
    print("=========================================")

# Calculate mean, standard deviation, and confidence intervals for each metric (original dataset)
for metric_name, scores in zip(
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [accuracy_scores_orig, precision_scores_orig, recall_scores_orig, f1_scores_orig]
):
    mean_score, std_dev, ci = confidence_interval(scores)
    results.append({
        'Model': 'Neural Network',
        'Dataset': 'Original',
        'Metric': metric_name,
        'Mean': mean_score,
        'Std Dev': std_dev,
        'CI Lower': ci[0],
        'CI Upper': ci[1]
    })

# Calculate mean, standard deviation, and confidence intervals for each metric (obfuscated dataset)
for metric_name, scores in zip(
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [accuracy_scores_obf, precision_scores_obf, recall_scores_obf, f1_scores_obf]
):
    mean_score, std_dev, ci = confidence_interval(scores)
    results.append({
        'Model': 'Neural Network',
        'Dataset': 'del',
        'Metric': metric_name,
        'Mean': mean_score,
        'Std Dev': std_dev,
        'CI Lower': ci[0],
        'CI Upper': ci[1]
    })

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("Neural/del_ci_d3.csv", index=False)

print("Metrics for original and obfuscated datasets saved to 'Neural")