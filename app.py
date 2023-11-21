from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import string
import pickle

app = Flask(__name__)

# Load the model from the pickle file
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorization from the pickle file
with open('vector.pkl', 'rb') as vector_file:
    vectorization = pickle.load(vector_file)

# Function to clear text data
def wordopt(text):
    # Your existing wordopt code here
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://S+|www.\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]'%re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['text']
        user_sample_test = pd.DataFrame({"text": [news_text]})
        user_sample_test['text'] = user_sample_test['text'].apply(wordopt)
        new_x_test = user_sample_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred = model.predict(new_xv_test)
        prediction = "It's A Fake News" if pred[0] == 0 else "It's A Real news"
        return render_template('index.html', prediction=prediction)

# Create an API endpoint for predictions
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if 'text' in data:
        news_text = data['text']
        user_sample_test = pd.DataFrame({"text": [news_text]})
        user_sample_test['text'] = user_sample_test['text'].apply(wordopt)
        new_x_test = user_sample_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred = model.predict(new_xv_test)
        prediction = "Fake News" if pred[0] == 0 else "Not A Fake News"
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    app.run(debug=True)
