import pandas as pd

# Load the datasets
df1 = pd.read_csv('constant_size_delay_frag/dataset/dcam_st_repadxor.csv')
df2 = pd.read_csv('constant_size_delay_frag/dataset/dswitch_st_repadxor.csv')
df3 = pd.read_csv('constant_size_delay_frag/dataset/hbridge_st_repadxor.csv')
df4 = pd.read_csv('constant_size_delay_frag/dataset/hswitch_st_repadxor.csv')
df5 = pd.read_csv('constant_size_delay_frag/dataset/light_st_repadxor.csv')
#df6 = pd.read_csv('constant_size_delay_frag/dataset/welcome_ori_current_time.csv')
#df7 = pd.read_csv('constant_size/dataset/welcome_ori.csv')


# Combine the datasets
combined_df = pd.concat([df1,df2,df3,df4,df5], ignore_index=True)

# Add a sequential numbering column
#combined_df['Row Number'] = range(1, len(combined_df) + 1)
combined_df.drop(['No.'], axis=1, inplace=True)

combined_df['Flags'] = combined_df['Flags'].str.replace('0x', '', regex=False)

#combined_df['Flags'] = combined_df['Flags'].replace('0c2', '2')

# Save to a new CSV file
combined_df.to_csv('constant_size_delay_frag/final/st_repadxor.csv', index=False)


