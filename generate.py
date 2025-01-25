import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your CSV file
data = pd.read_csv('your_data.csv')

# Create LabelEncoders for categorical variables
le_race = LabelEncoder()
le_gender = LabelEncoder()
le_case_type = LabelEncoder()

# Encoding categorical columns
data['race'] = le_race.fit_transform(data['race'])
data['gender'] = le_gender.fit_transform(data['gender'])
data['case_type'] = le_case_type.fit_transform(data['case_type'])

# Define your target variable 'outcome'
X = data[['race', 'gender', 'age', 'case_type']]
y = data['outcome']

# Train your model (for example RandomForest)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Fit the model
model.fit(X, y)

# Save encoders and the model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le_race, open('le_race.pkl', 'wb'))
pickle.dump(le_gender, open('le_gender.pkl', 'wb'))
pickle.dump(le_case_type, open('le_case_type.pkl', 'wb'))

print("Model and LabelEncoders have been saved successfully.")
