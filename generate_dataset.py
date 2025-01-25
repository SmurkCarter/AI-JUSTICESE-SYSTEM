import pandas as pd
import numpy as np

# Define the number of rows (cases)
num_cases = 1000

# Generate random data
data = {
    'race': np.random.randint(0, 2, num_cases),  # 0 for non-minority, 1 for minority
    'gender': np.random.randint(0, 2, num_cases),  # 0 for male, 1 for female
    'age': np.random.randint(18, 65, num_cases),  # Random age between 18 and 65
    'case_type': np.random.randint(0, 4, num_cases),  # Case type (0, 1, 2, or 3)
    'outcome': np.random.randint(0, 2, num_cases)  # 0 for negative, 1 for positive
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('legal_cases_dataset.csv', index=False)

print("Dataset saved as legal_cases_dataset.csv")
