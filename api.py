import pickle
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Preprocessing Setup ---
pt = PorterStemmer()

# Hardcoded English stopwords (no NLTK downloads required)
stopWords = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now"
}

# --- Model & Vectorizer Loading ---
with open("vectorizer.pkl", "rb") as vf:
    tfidf = pickle.load(vf)
with open("model.pkl", "rb") as mf:
    mnb = pickle.load(mf)

# --- Preprocessing Function ---
def data_preprocess(text: str) -> str:
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    cleaned = [w for w in tokens if w.isalnum() and w not in stopWords]
    stemmed = [pt.stem(w) for w in cleaned]
    return " ".join(stemmed)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "API is alive and ready to predict."
    })

@app.route("/predict", methods=["POST"])
def predict_spam():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    input_sms = data["text"]

    try:
        transformed_sms = data_preprocess(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = mnb.predict(vector_input)[0]
        label = "Spam" if result == 1 else "Non-spam"

        return jsonify({
            "prediction_label": label,
            "prediction_value": int(result),
            "text": input_sms
        })

    except Exception as e:
        print(f"Prediction failed for input '{input_sms}': {e}")
        return jsonify({"error": "Internal prediction error."}), 500

# --- Run Local Server ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
