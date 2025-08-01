import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

# Load the original dataset
original_dataset_path = "AD_dataset/st_original.csv"
original_dataset = pd.read_csv(original_dataset_path)

# Preprocess the original dataset
X_orig = original_dataset.drop(['Label','Flags'], axis=1)
y_orig = original_dataset['Label']

# Load the new (obfuscated) dataset
new_dataset_path = 'AD_dataset/st_con.csv'
new_dataset = pd.read_csv(new_dataset_path)
X_new = new_dataset.drop(['Label','Flags'], axis=1)
y_new = new_dataset['Label']

# Normalize the data
scaler_orig = StandardScaler()
X_orig_scaled = scaler_orig.fit_transform(X_orig)
X_new_scaled = scaler_orig.transform(X_new)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=53, random_state=30),            
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=30),
    "Gradient Boosting Machine": GradientBoostingClassifier(random_state=30)
}

# 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results dictionary to store the metrics for both datasets
results = []

# Function to calculate confidence intervals
def confidence_interval(scores):
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    
    # Handle the case where the standard deviation is zero by setting a small near-zero value
    if std_dev == 0:
        std_dev = 1e-6  # Set to a small near-zero value to avoid zero standard deviation

    # Calculate the confidence interval using the adjusted std_dev
    ci = stats.t.interval(0.95, len(scores) - 1, loc=mean_score, scale=std_dev / np.sqrt(len(scores)))
    
    return mean_score, std_dev, ci

# Loop through each model
for model_name, model in models.items():
    print(f"**************{model_name}********************")
    
    # Store metrics for each fold
    accuracy_scores_orig = []
    precision_scores_orig = []
    recall_scores_orig = []
    f1_scores_orig = []

    accuracy_scores_new = []
    precision_scores_new = []
    recall_scores_new = []
    f1_scores_new = []
    
    for train_index, test_index in skf.split(X_orig_scaled, y_orig):
        # Original dataset train/test split
        X_train_fold, X_test_fold = X_orig_scaled[train_index], X_orig_scaled[test_index]
        y_train_fold, y_test_fold = y_orig[train_index], y_orig[test_index]

        # Train the model on the original dataset's training set
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on the original dataset
        y_pred_fold_orig = model.predict(X_test_fold)
        accuracy_scores_orig.append(accuracy_score(y_test_fold, y_pred_fold_orig))
        precision_scores_orig.append(precision_score(y_test_fold, y_pred_fold_orig, average='macro'))  # Using macro avg
        recall_scores_orig.append(recall_score(y_test_fold, y_pred_fold_orig, average='macro'))        # Using macro avg
        f1_scores_orig.append(f1_score(y_test_fold, y_pred_fold_orig, average='macro'))                # Using macro avg

        # Evaluate the model on the new (obfuscated) dataset's test set
        y_pred_fold_new = model.predict(X_new_scaled)
        accuracy_scores_new.append(accuracy_score(y_new, y_pred_fold_new))
        precision_scores_new.append(precision_score(y_new, y_pred_fold_new, average='macro'))
        recall_scores_new.append(recall_score(y_new, y_pred_fold_new, average='macro'))
        f1_scores_new.append(f1_score(y_new, y_pred_fold_new, average='macro'))
    
    # Calculate mean, std dev, and confidence intervals for each metric (original dataset)
    for metric_name, scores in zip(
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        [accuracy_scores_orig, precision_scores_orig, recall_scores_orig, f1_scores_orig]
    ):
        mean_score, std_dev, ci = confidence_interval(scores)
        results.append({
            'Model': model_name,
            'Dataset': 'Original',
            'Metric': metric_name,
            'Mean': mean_score,
            'Std Dev': std_dev,
            'CI Lower': ci[0],
            'CI Upper': ci[1]
        })

    # Calculate mean, std dev, and confidence intervals for each metric (new obfuscated dataset)
    for metric_name, scores in zip(
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        [accuracy_scores_new, precision_scores_new, recall_scores_new, f1_scores_new]
    ):
        mean_score, std_dev, ci = confidence_interval(scores)
        results.append({
            'Model': model_name,
            'Dataset': 'con',
            'Metric': metric_name,
            'Mean': mean_score,
            'Std Dev': std_dev,
            'CI Lower': ci[0],
            'CI Upper': ci[1]
        })

    print("=" * 80)

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("Confidence_Interval/Sentinel/con_ci.csv", index=False)
print("Results saved to model_metrics_with_ci_obfuscated.csv")
