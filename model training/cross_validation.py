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
original_dataset_path = "AD_dataset/original_dataset.csv"
original_dataset = pd.read_csv(original_dataset_path)

# Preprocess the original dataset
X_orig = original_dataset.drop(['Label', 'Flags'], axis=1)
y_orig = original_dataset['Label']

# Normalize the data
scaler_orig = StandardScaler()
X_orig_scaled = scaler_orig.fit_transform(X_orig)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=53, random_state=30),            
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=30),
    "Gradient Boosting Machine": GradientBoostingClassifier(random_state=30)
}

# 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results dictionary to store the metrics
results = []

# Function to calculate confidence intervals
def confidence_interval(scores):
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    if len(scores) > 1:
        ci = stats.t.interval(0.95, len(scores) - 1, loc=mean_score, scale=std_dev / np.sqrt(len(scores)))
    else:
        ci = (mean_score, mean_score)  # If there's only one value, CI can't be calculated properly
    return mean_score, std_dev, ci

# Loop through each model
for model_name, model in models.items():
    print(f"**************{model_name}********************")
    
    # Store metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in skf.split(X_orig_scaled, y_orig):
        X_train_fold, X_test_fold = X_orig_scaled[train_index], X_orig_scaled[test_index]
        y_train_fold, y_test_fold = y_orig[train_index], y_orig[test_index]
        
        # Train the model on the fold
        model.fit(X_train_fold, y_train_fold)
        
        # Make predictions
        y_pred_fold = model.predict(X_test_fold)
        
        # Calculate metrics for this fold
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
        precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='weighted'))
        recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='weighted'))
        f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted'))
    
    # Calculate mean, std dev, and confidence intervals for each metric
    for metric_name, scores in zip(
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        [accuracy_scores, precision_scores, recall_scores, f1_scores]
    ):
        mean_score, std_dev, ci = confidence_interval(scores)
        results.append({
            'Model': model_name,
            'Metric': metric_name,
            'Mean': mean_score,
            'Std Dev': std_dev,
            'CI Lower': ci[0],
            'CI Upper': ci[1]
        })

    print("=" * 80)

# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("model_metrics_with_ci (UNSW).csv", index=False)
print("Results saved to model_metrics_with_ci.csv")
