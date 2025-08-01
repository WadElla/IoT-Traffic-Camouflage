import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('dataset/repadded.csv')

# Setting a random seed for reproducibility
random_seed = 42

# Splitting the dataset into three equal parts
part1, temp = train_test_split(df, test_size=2/3, stratify=df['Label'], random_state=random_seed)
part2, part3 = train_test_split(temp, test_size=1/2, stratify=temp['Label'], random_state=random_seed)


X = part1.drop(['Label'], axis=1)
y = part1['Label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

combined_df1 = pd.concat([X_train, y_train], axis=1) # vertical
combined_df2 = pd.concat([X_test, y_test], axis=1)


X1 = part2.drop(['Label'], axis=1)
y1 = part2['Label']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=0.20, random_state=42)

combined_df3 = pd.concat([X_train1, y_train1], axis=1)
combined_df4 = pd.concat([X_test1, y_test1], axis=1)



X2 = part3.drop(['Label'], axis=1)
y2 = part3['Label']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=0.20, random_state=42)

combined_df5 = pd.concat([X_train2, y_train2], axis=1)
combined_df6 = pd.concat([X_test2, y_test2], axis=1)


combined_df_f1 = pd.concat([combined_df1, combined_df3, combined_df5], ignore_index=True) #horizontal

combined_df_f2 = pd.concat([combined_df2, combined_df4, combined_df6], ignore_index=True)


combined_df_f1.to_csv('dataset_split/repadded_train.csv', index=False)

combined_df_f2.to_csv('dataset_split/repadded_test.csv', index=False)








# Optionally, save these to new CSV files
#part1.to_csv('Divide/part1_padded.csv', index=False)
#part2.to_csv('Divide/part2_padded.csv', index=False)
#part3.to_csv('Divide/part3_padded.csv', index=False)
