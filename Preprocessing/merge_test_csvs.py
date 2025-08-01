import pandas as pd

# Load the datasets
df1 = pd.read_csv('test/test_data_pad.csv')
df2 = pd.read_csv('test/test_data2_pad.csv')
df3 = pd.read_csv('test/test_data3_pad.csv')

# Combine the datasets
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save to a new CSV file
combined_df.to_csv('test/test_final_pad.csv', index=False)
