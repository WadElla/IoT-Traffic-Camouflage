import pandas as pd

# Load the datasets
df1 = pd.read_csv('AD_final/amazonplug.csv')
df2 = pd.read_csv('AD_final/baby.csv')
df3 = pd.read_csv('AD_final/echo.csv')
df4 = pd.read_csv('AD_final/ecobee.csv')
df5 = pd.read_csv('AD_final/kettle.csv')
df6 = pd.read_csv('AD_final/plugwm.csv')
df7 = pd.read_csv('AD_final/speaker.csv')

# Combine the datasets
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)

# Drop the 'No.' column if it exists
if 'No.' in combined_df.columns:
    combined_df.drop(['No.'], axis=1, inplace=True)

# Remove the '0x' prefix from the 'Flags' column if it exists
if 'Flags' in combined_df.columns:
    combined_df['Flags'] = combined_df['Flags'].str.replace('0x', '', regex=False)

# Check for non-numeric values
for column in combined_df.columns:
    if combined_df[column].dtype == object:
        print(f"Checking column {column} for non-numeric values...")
        non_numeric = combined_df[column].apply(lambda x: not x.isnumeric() if isinstance(x, str) else False)
        if non_numeric.any():
            print(f"Non-numeric values found in column {column}:")
            print(combined_df[non_numeric])

# Save to a new CSV file
combined_df.to_csv('AD_final/AD_original.csv', index=False)
