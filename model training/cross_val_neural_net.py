import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
import tensorflow as tf

# Load and prepare the dataset
file_path = 'AD_dataset/ad_ori.csv'
dataset = pd.read_csv(file_path)

# Prepare features and labels
X = dataset.drop(['Label','Flags'], axis=1)
y = dataset['Label']

# Normalize the features with StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels with OneHotEncoder
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

# Define stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store results
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Function to calculate confidence intervals
def confidence_interval(scores):
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    if len(scores) > 1:
        ci = stats.t.interval(0.95, len(scores) - 1, loc=mean_score, scale=std_dev / np.sqrt(len(scores)))
    else:
        ci = (mean_score, mean_score)  # In case of only one value, CI can't be calculated
    return mean_score, std_dev, ci

# Loop through each fold in the StratifiedKFold
for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y)):
    print(f"************** Fold {fold + 1} **************")
    
    # Split data into training and testing for this fold
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]
    
    # Define and compile the model
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
    
    # Train the model on this fold
    model.fit(X_train_fold, y_train_fold, validation_split=0.03, epochs=295, batch_size=32, verbose=0)
    
    # Predict and evaluate on the test fold
    y_pred_fold = model.predict(X_test_fold)
    y_pred_classes_fold = np.argmax(y_pred_fold, axis=1)
    y_true_classes_fold = np.argmax(y_test_fold, axis=1)
    
    # Calculate metrics for this fold
    accuracy = accuracy_score(y_true_classes_fold, y_pred_classes_fold)
    precision = precision_score(y_true_classes_fold, y_pred_classes_fold, average='weighted')
    recall = recall_score(y_true_classes_fold, y_pred_classes_fold, average='weighted')
    f1 = f1_score(y_true_classes_fold, y_pred_classes_fold, average='weighted')
    
    # Store metrics for each fold
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print("=========================================")

# Calculate mean, standard deviation, and confidence interval for each metric
metrics_results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Mean': [
        np.mean(accuracy_scores),
        np.mean(precision_scores),
        np.mean(recall_scores),
        np.mean(f1_scores)
    ],
    'Standard Deviation': [
        np.std(accuracy_scores),
        np.std(precision_scores),
        np.std(recall_scores),
        np.std(f1_scores)
    ],
    'Confidence Interval Lower': [
        confidence_interval(accuracy_scores)[2][0],
        confidence_interval(precision_scores)[2][0],
        confidence_interval(recall_scores)[2][0],
        confidence_interval(f1_scores)[2][0]
    ],
    'Confidence Interval Upper': [
        confidence_interval(accuracy_scores)[2][1],
        confidence_interval(precision_scores)[2][1],
        confidence_interval(recall_scores)[2][1],
        confidence_interval(f1_scores)[2][1]
    ]
}

# Convert results to a DataFrame
metrics_df = pd.DataFrame(metrics_results)

# Save the results to a CSV file
metrics_df.to_csv("model_metrics_ci(IoT-AD (neural net)).csv", index=False)
print("Cross-validation metrics saved to model_cv_metrics_results.csv")
