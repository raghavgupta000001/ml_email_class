import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Preprocessing Setup ---
# Initialize stemmer and ensure stopwords are loaded
pt = PorterStemmer()

# --- Model & Vectorizer Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECT_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

try:
    # Load assets
    tfidf = pickle.load(open(VECT_PATH, 'rb'))
    mnb = pickle.load(open(MODEL_PATH, 'rb'))
    
    # Load NLTK data (Crucial for deployment builds)
    # These downloads will run during Render's build process.
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    stopWords = stopwords.words("english")
    
    print("Assets loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load assets or NLTK data: {e}")
    tfidf = None
    mnb = None
    stopWords = [] # Prevent runtime error if model fails

# --- Preprocessing Function ---
def data_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum() and i not in stopWords and i not in string.punctuation]
    y = [pt.stem(i) for i in y]
    return " ".join(y)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enables communication with your Vercel frontend

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_spam():
    if mnb is None or tfidf is None:
        return jsonify({"error": "Model failed to load on server startup."}), 500
    
    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400
        
    input_sms = data['text']
    
    try:
        # 1. Preprocess input
        transformed_sms = data_preprocess(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = mnb.predict(vector_input)[0]
        
        label = "Spam" if result == 1 else "Non-spam"
        
        return jsonify({
            "prediction_label": label,
            "prediction_value": int(result),
            "text": input_sms
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500