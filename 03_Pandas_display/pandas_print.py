import pandas as pd

# Create CSV file
with open('sample.csv', 'w') as f:
    f.write('name,age,place\n')
    f.write('Ada,24,Tokyo\n')
    f.write('Bill,42,Osaka\n')
    f.write('Claire,19,Kyoto\n')

# Read CSV file with pandas and convert to data frame
df = pd.read_csv('sample.csv')

# Show data frame
print(df)

# Save data frame to CSV file
df.to_csv('output.csv', index=False)
