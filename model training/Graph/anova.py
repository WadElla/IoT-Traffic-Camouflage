import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('AD_dataset/original_dataset.csv')

# Separate the features and the target variable
X = data.drop(columns=['Label'])  # All columns except 'Label'
y = data['Label']  # The 'Label' column

# Perform feature selection
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Get the scores and feature names
scores = selector.scores_
feature_names = X.columns

# Create a DataFrame to sort by scores
features_df = pd.DataFrame({'Feature': feature_names, 'Score': scores})
features_df = features_df.sort_values(by='Score', ascending=False)

# Extract sorted feature names and scores
sorted_feature_names = features_df['Feature']
sorted_scores = features_df['Score']

# Create a bar plot of the scores
plt.figure(figsize=(14, 8))  # Adjust figure size as needed
bars = plt.bar(range(len(sorted_scores)), sorted_scores, align='center', alpha=0.7, color='b', hatch='//')
plt.xticks(range(len(sorted_scores)), sorted_feature_names, rotation=45, ha='right')
plt.xlabel('Feature Names')
plt.ylabel('F-Scores')
plt.title('ANOVA Feature Selection Scores', pad=20)  # Adjust the padding to avoid overlap
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add score labels on top of the bars
for bar, score in zip(bars, sorted_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.05 * max(sorted_scores), f'{score:.2f}', ha='center', va='bottom', fontsize=10, rotation=90, color='black')

# Save the plot as a high-quality PDF
plt.tight_layout(pad=2.0)  # Adjust padding
plt.savefig('Graph/anova.pdf', format='pdf', dpi=300)
plt.show()
