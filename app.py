from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import chardet
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_race = pickle.load(open('le_race.pkl', 'rb'))
le_gender = pickle.load(open('le_gender.pkl', 'rb'))
le_case_type = pickle.load(open('le_case_type.pkl', 'rb'))

# Function to scrape legal information
def scrape_legal_info(query):
    try:
        # Formulate the Google search query URL
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}+site:kenyalaw.org"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the first result (e.g., title, snippet, link)
        result = soup.find('h3')
        if result:
            title = result.text
            snippet = result.find_next('span').text
            link = result.find_parent('a')['href']
            return {
                'title': title,
                'snippet': snippet,
                'link': link
            }
        else:
            return None
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_page')
def predict_page():
    return render_template('predict_page.html')


@app.route('/legal_guidance')
def legal_guidance_page():
    return render_template('legal_guidance.html')


@app.route('/get_guidance', methods=['POST'])
def get_guidance():
    try:
        # Attempt to get the question from form data or JSON
        question = request.form.get('question', '').lower()

        # If no form data, fall back to JSON
        if not question:
            data = request.get_json()
            if not data:
                return jsonify({
                    'message': 'Invalid or missing JSON data. Please ensure you are sending a valid JSON payload.'
                }), 400
            question = data.get('question', '').lower()

        # Call the scraping function to get answers
        search_results = scrape_legal_info(question)

        if search_results:
            response_message = f"Title: {search_results['title']}\n\nSnippet: {search_results['snippet']}\n\nLink: {search_results['link']}"
        else:
            response_message = "Sorry, I couldn't find relevant information."

        return jsonify({'message': response_message})

    except Exception as e:
        return jsonify({'message': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a case file is uploaded
        if 'case_file' in request.files:
            case_file = request.files['case_file']
            if case_file.filename != '':
                # Detect the file encoding using chardet
                case_file_content = case_file.read()  # Read file content
                result = chardet.detect(case_file_content)
                encoding = result['encoding']

                # Use detected encoding to read the file
                case_file.seek(0)  # Reset file pointer
                case_data = pd.read_csv(case_file, encoding=encoding)

                # Get case number input from user
                case_number = request.form['case_number']
                case_info = case_data[case_data['case_number'] == int(case_number)]

                if not case_info.empty:
                    # Ensure correct data type
                    input_data = case_info[['race', 'gender', 'age', 'case_type']].values
                    input_data = input_data.astype(str)  # Convert all to string
                    prediction = model.predict(input_data)
                    outcome_msg = "Outcome: Guilty" if prediction[0] == 1 else "Outcome: Not Guilty"
                    return jsonify({'message': outcome_msg})
                else:
                    return jsonify({'message': 'Case not found!'})

        # If no file is uploaded, use manual form inputs
        race = request.form['race']
        gender = request.form['gender']
        age = request.form['age']
        case_type = request.form['case_type']

        # Validate inputs
        if race not in le_race.classes_:
            return jsonify({'message': 'Invalid race input. Please choose from: ' + ', '.join(le_race.classes_)})
        if gender not in le_gender.classes_:
            return jsonify({'message': 'Invalid gender input. Please choose from: ' + ', '.join(le_gender.classes_)})
        if case_type not in le_case_type.classes_:
            return jsonify(
                {'message': 'Invalid case type input. Please choose from: ' + ', '.join(le_case_type.classes_)})

        # Label encode inputs
        race_encoded = le_race.transform([race])[0]
        gender_encoded = le_gender.transform([gender])[0]
        case_type_encoded = le_case_type.transform([case_type])[0]

        # Prepare input for model, ensuring correct types
        input_data = [[race_encoded, gender_encoded, int(age), case_type_encoded]]
        prediction = model.predict(np.array(input_data, dtype=object))  # Ensure all data is treated as objects

        # Simple outcome message
        outcome_msg = "Outcome: Guilty" if prediction[0] == 1 else "Outcome: Not Guilty"

        # Return the result in a pop-up
        return jsonify({'message': outcome_msg})

    except Exception as e:
        return jsonify({'message': str(e)})


@app.route('/check_bias', methods=['POST'])
def check_bias():
    try:
        # Input scenario from the user
        scenario = request.form['scenario'].lower()

        # Keywords to detect potential bias
        bias_keywords = {
            'black': 1, 'white': 1, 'race': 1, 'gender': 0.5,
            'arrested': 1, 'detained': 1, 'discriminated': 1,
            'disability': 1, 'poor': 1, 'rich': 1, 'homeless': 1,
            'unemployed': 1, 'low income': 1, 'LGBTQ': 1, 'minority': 1
        }

        # Check for bias in the scenario
        is_biased = any(keyword in scenario for keyword in bias_keywords)

        # Return a simple pop-up message for the user
        if is_biased:
            return jsonify({'message': 'Bias Detected'})
        else:
            return jsonify({'message': 'No Bias Detected'})

    except Exception as e:
        return jsonify({'message': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
