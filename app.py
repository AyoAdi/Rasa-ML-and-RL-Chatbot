from flask import Flask, render_template, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RASA_API_URL = 'http://localhost:5002/webhooks/rest/webhook'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    sender_id = data.get('sender_id')

    if not user_message or not sender_id:
        return jsonify({"reply": "No message or sender_id received :<."}), 400

    try:
        response = requests.post(
            RASA_API_URL,
            json={"sender": sender_id, "message": user_message},
            timeout=5
        )
        response.raise_for_status()
        rasa_data = response.json()

        bot_reply = "Sorry, Didn't get ya."
        if rasa_data and isinstance(rasa_data, list):
            bot_texts = [r.get('text') for r in rasa_data if 'text' in r]
            if bot_texts:
                bot_reply = " ".join(bot_texts)

        return jsonify({"reply": bot_reply})

    except requests.exceptions.RequestException as e:
        return jsonify({"reply": f"Error :<: {e}"}), 500

if __name__ == '__main__':
    app.run(port=5005, debug=True)