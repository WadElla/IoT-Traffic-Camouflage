import pandas as pd

def transform_csv(file_list, output_list):
    # Loop through all input files
    for idx, input_file in enumerate(file_list):
        # Read the input CSV file
        df = pd.read_csv(input_file)

        # Convert Mean, CI Lower, and CI Upper to percentages (2 decimal places)
        df['Mean'] = (df['Mean'] * 100).round(2)
        df['CI Lower'] = (df['CI Lower'] * 100).round(2)
        df['CI Upper'] = (df['CI Upper'] * 100).round(2)

        # Convert Std to five decimal places (do not convert to percentages)
        df['Std Dev'] = df['Std Dev'].round(5)

        # Save the transformed CSV file
        output_file = output_list[idx]
        df.to_csv(output_file, index=False)
        print(f"File '{output_file}' has been saved.")

# Example usage
input_files = ['Neural/AD/con_ci_d1.csv', 'Neural/AD/del_ci_d1.csv', 'Neural/AD/frag_ci_d1.csv', 'Neural/AD/pad_ci_d1.csv', 'Neural/AD/padshift_ci_d1.csv', 'Neural/AD/padxor_ci_d1.csv']  # List your input CSV files here
output_files = ['Conf/AD/con_d1.csv', 'Conf/AD/del_d1.csv', 'Conf/AD/frag_d1.csv', 'Conf/AD/pad_d1.csv','Conf/AD/padshift_d1.csv','Conf/AD/padxor_d1.csv']  # List the corresponding output CSV file names

transform_csv(input_files, output_files)
