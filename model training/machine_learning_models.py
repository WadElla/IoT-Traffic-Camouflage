import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load the original dataset
            # original_dataset
            # ori_dset_cutime
original_dataset_path = "AD_dataset/original_dataset.csv"
original_dataset = pd.read_csv(original_dataset_path)

#original_dataset = original_dataset.fillna(0)

# Preprocess the original dataset
X_orig = original_dataset.drop(['Label'], axis=1)
y_orig = original_dataset['Label']

# Split the original dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.20, stratify=y_orig, random_state=42)

# Normalize the training data
scaler_orig = StandardScaler()
X_train_scaled = scaler_orig.fit_transform(X_train)
X_test_scaled = scaler_orig.transform(X_test)

# Define and train models    # n_estimators=53, random_state=30
models = {
    "Random Forest": RandomForestClassifier(n_estimators=53, random_state=30),            
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=30),
    "Gradient Boosting Machine": GradientBoostingClassifier(random_state=30),
    #"Support Vector Machine": SVC(kernel='linear', random_state=30)
}

# Load the new dataset for testing
new_dataset_path = 'AD_dataset/frag_nodel.csv'
new_dataset = pd.read_csv(new_dataset_path)

#new_dataset = new_dataset.fillna(0)

# Preprocess the new dataset (using the scaler fitted on the original dataset)
X_new = new_dataset.drop(['Label'], axis=1)
y_new = new_dataset['Label']
X_new_scaled = scaler_orig.transform(X_new)

for model_name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model on the original dataset's test set
    y_pred_test = model.predict(X_test_scaled)
    print(f"**************{model_name}********************")
    print(f'Accuracy on the original dataset test set: {accuracy_score(y_test, y_pred_test)}')
    print()
    print(classification_report(y_test, y_pred_test))
    
    # Test the model on the new dataset
    y_new_pred = model.predict(X_new_scaled)
    
    # Evaluate the model's performance on the new dataset
    print(f'Accuracy on the new dataset: {accuracy_score(y_new, y_new_pred)}')
    print()
    print(classification_report(y_new, y_new_pred))
    
    # Print feature importance for applicable models
    if model_name in ["Random Forest", "Gradient Boosting Machine", "Decision Tree"]:
        feature_importances = model.feature_importances_
        feature_importances_percent = 100.0 * (feature_importances / feature_importances.sum())
        feature_names = X_orig.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_percent})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print(f'Feature importances for {model_name} (in percentage):')
        print(importance_df)
    
    print("="*80)
