import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys # Added for error handling

# --- Preprocessing Setup ---
# Initialize stemmer
pt = PorterStemmer()
stopWords = [] # Initialize as empty, will be loaded later

# --- Model & Vectorizer Loading ---
# Determine paths relative to the current script's directory
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECT_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

tfidf = None
mnb = None
asset_load_success = False

try:
    # Load assets
    with open(VECT_PATH, 'rb') as vf:
        tfidf = pickle.load(vf)
    with open(MODEL_PATH, 'rb') as mf:
        mnb = pickle.load(mf)
    
    # Load NLTK data (Crucial for deployment builds)
    # This ensures the necessary data is available for the stopwords object
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    global stopWords
    stopWords = stopwords.words("english")
    
    print("Assets and NLTK data loaded successfully!")
    asset_load_success = True
    
except Exception as e:
    # Print the error for Render logs and fail gracefully
    print(f"CRITICAL ERROR: Failed to load model or NLTK data: {e}", file=sys.stderr)
    asset_load_success = False

# --- Preprocessing Function ---
def data_preprocess(text):
    global stopWords
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric, stopwords, and punctuation
    y = [i for i in text if i.isalnum() and i not in stopWords and i not in string.punctuation]
    
    # Stemming
    y = [pt.stem(i) for i in y]
    return " ".join(y)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enables communication with your Vercel frontend

# --- Health Check Endpoint (For debugging 404s) ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "success",
        "message": "API is alive and ready to predict.",
        "model_loaded": asset_load_success
    })

# --- API Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_spam():
    # Check if models loaded successfully on startup
    if not asset_load_success:
        return jsonify({"error": "Server failed to load ML assets on startup."}), 500
    
    data = request.get_json(force=True, silent=True)
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body. Must be JSON."}), 400
        
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
        # Log the internal prediction error
        print(f"Prediction failed for input '{input_sms}': {e}", file=sys.stderr)
        return jsonify({"error": f"Internal prediction error."}), 500

if __name__ == '__main__':
    # Local development server
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))