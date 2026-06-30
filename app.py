from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import pandas as pd

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading dataset...")
df = pd.read_excel('data/Mahabharata_Character_Dataset_300.xlsx')
data = df.to_dict(orient='records')

dataset_texts = [
    str(entry["Emotional State"]) + " " +
    str(entry["Life Domain"]) + " " +
    str(entry["Teaching"]) + " " +
    str(entry["Advice"])
    for entry in data
]

print("Creating embeddings...")
dataset_embeddings = model.encode(dataset_texts)

print("System Ready!")

def find_best_match(user_input):
    user_embedding = model.encode(user_input)
    similarities = util.cos_sim(user_embedding, dataset_embeddings)
    best_index = similarities.argmax()
    return data[best_index]

def generate_advice(user_input, teaching):

    return f"""
You mentioned: "{user_input}"

{teaching}

Try to approach this situation calmly and take small, thoughtful steps. With patience and clarity, you will find the right direction.
"""


# 🔥 API route
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message")

    result = find_best_match(user_input)
    generated_advice = generate_advice(user_input, result["Teaching"])

    return jsonify({
        "character": result["Character"],
        "emotion": result["Emotional State"],
        "domain": result["Life Domain"],
        "teaching": result["Teaching"],
        "advice": generated_advice
    })

@app.route("/")
def home():
    return "Saarthi Backend Running ✅"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)