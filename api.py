import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

# --- Preprocessing Setup ---
pt = PorterStemmer()

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stopWords = stopwords.words("english")

# --- Model & Vectorizer Loading ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECT_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

tfidf = None
mnb = None
asset_load_success = False

try:
    with open(VECT_PATH, 'rb') as vf:
        tfidf = pickle.load(vf)
    with open(MODEL_PATH, 'rb') as mf:
        mnb = pickle.load(mf)

    print("✅ Assets and NLTK data loaded successfully!")
    asset_load_success = True

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load model/vectorizer: {e}", file=sys.stderr)
    asset_load_success = False


# --- Preprocessing Function ---
def data_preprocess(text: str) -> str:
    text = text.lower()
    from nltk.tokenize import wordpunct_tokenize
    tokens = wordpunct_tokenize(text)


    # Keep only alphanumeric, remove stopwords & punctuation
    cleaned = [w for w in tokens if w.isalnum() and w not in stopWords and w not in string.punctuation]

    # Apply stemming
    stemmed = [pt.stem(w) for w in cleaned]
    return " ".join(stemmed)


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)


# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "success",
        "message": "API is alive and ready to predict.",
        "model_loaded": asset_load_success
    })


# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_spam():
    if not asset_load_success:
        return jsonify({"error": "Server failed to load ML assets on startup."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body. Must be JSON."}), 400

    input_sms = data['text']

    try:
        # 1. Preprocess
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
        print(f"Prediction failed for input '{input_sms}': {e}", file=sys.stderr)
        return jsonify({"error": "Internal prediction error."}), 500


# --- Run Local Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
