
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, render_template, request

# Load dataset
data = pd.read_csv('legal_cases_dataset.csv')

# Basic preprocessing
label_encoder = LabelEncoder()
data['race'] = label_encoder.fit_transform(data['race'])
data['gender'] = label_encoder.fit_transform(data['gender'])
data['case_type'] = label_encoder.fit_transform(data['case_type'])

# Split dataset into features and labels
X = data[['race', 'gender', 'age', 'case_type']]
y = data['outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix to understand prediction results
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Define bias detection function
def detect_bias(predictions, group_labels, sensitive_feature):
    if len(predictions) == 1:  # For single prediction
        return "Bias detection not applicable for single input."

    group_0 = predictions[group_labels[sensitive_feature] == 0]
    group_1 = predictions[group_labels[sensitive_feature] == 1]
    rate_0 = sum(group_0) / len(group_0) if len(group_0) > 0 else 0
    rate_1 = sum(group_1) / len(group_1) if len(group_1) > 0 else 0
    bias_detected = abs(rate_0 - rate_1)
    return bias_detected

# Flask app for user input and prediction
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    race = int(request.form['race'])
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    case_type = int(request.form['case_type'])

    # Input to the model
    input_data = np.array([[race, gender, age, case_type]])

    # Make prediction
    prediction = model.predict(input_data)

    # Detect bias
    bias = detect_bias(prediction, X_test, 'race')

    # Check if bias is a string or a float
    if isinstance(bias, str):
        return f'Predicted outcome: {prediction[0]}, Bias detected: {bias}'
    else:
        return f'Predicted outcome: {prediction[0]}, Bias detected: {bias:.2f}'

if __name__ == '__main__':
    app.run(debug=True)
@app.route('/legal_info')
def legal_info():
    return render_template('legal_guidance.html')
