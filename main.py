import openai
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Charger le modèle et le vectorizer
model = joblib.load('final_model_rf.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Clé API OpenAI
openai.api_key = "Ma cle est privée"

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.get_json()
    symptoms = data.get('message', '')

    # Vérifier si les symptômes sont fournis
    if not symptoms:
        return jsonify({'error': 'Aucun symptôme fourni'}), 400

    # Transformer les symptômes en vecteur TF-IDF
    input_vector = tfidf.transform([symptoms])

    # Faire une prédiction avec le modèle
    prediction = model.predict(input_vector)[0]

    # Obtenir un diagnostic de ChatGPT en tant que professionnel de la médecine
    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Vous êtes un professionnel de la médecine."},
            {"role": "user", "content": f"Veuillez fournir un diagnostic basé sur les symptômes suivants : {symptoms}."}
        ]
    )

    chatgpt_diagnosis = chatgpt_response['choices'][0]['message']['content']

    # Retourner la prédiction du modèle et le diagnostic de ChatGPT
    return jsonify({
        'predicted_disease': prediction,
        'chatgpt_diagnosis': chatgpt_diagnosis
    })

if __name__ == '__main__':
    app.run(debug=True)
