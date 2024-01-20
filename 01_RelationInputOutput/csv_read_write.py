import pandas as pd

# Read the input.csv file
input_data = pd.read_csv('input.csv')

# Process the data (e.g., increase age by 10)
input_data['age'] = input_data['age'] + 10

# Write the processed data to the output.csv file with UTF-8 encoding and without the index
input_data.to_csv('output.csv', encoding="utf-8-sig", index=False)

# Check the content of the output file
print("outputfile:")
print(pd.read_csv('output.csv'))


