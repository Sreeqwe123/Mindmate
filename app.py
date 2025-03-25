from flask import Flask, request, jsonify
import torch
import random
import json
from nltk.stem import PorterStemmer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents
with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE, map_location=device)  # Ensures compatibility across devices
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "MindMate"
stemmer = PorterStemmer()

# Fallback responses when bot is unsure
fallback_responses = [
    "I'm not sure I understand. Can you rephrase that?",
    "That’s an interesting question! Can you tell me more?",
    "I’d love to help! Can you give me more details?",
    "I might not have the perfect answer, but I’m here to listen.",
    "Could you clarify what you mean? I want to help!"
]

chat_sessions = {}  # Store chat history

def get_best_matching_intent(user_input):
    """Find the closest matching intent using word overlap."""
    user_words = [stemmer.stem(word.lower()) for word in tokenize(user_input)]
    
    best_match = None
    highest_match_score = 0.0

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_words = [stemmer.stem(word.lower()) for word in tokenize(pattern)]
            match_score = len(set(user_words) & set(pattern_words)) / len(set(pattern_words))  

            if match_score > highest_match_score:
                highest_match_score = match_score
                best_match = intent["tag"]

    return best_match if highest_match_score > 0.4 else None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user_id")
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Initialize chat session if not exists
    if user_id not in chat_sessions:
        chat_sessions[user_id] = []

    chat_sessions[user_id].append(f"You: {user_message}")  # Store user input

    # Process message with the trained chatbot model
    tokenized_sentence = tokenize(user_message)
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    prob = torch.softmax(output, dim=1)[0][predicted.item()]

    if prob.item() >= 0.75:
        tag = tags[predicted.item()]
    else:
        tag = get_best_matching_intent(user_message)

    response = random.choice(fallback_responses)
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            break

    chat_sessions[user_id].append(f"{bot_name}: {response}")

    return jsonify({"bot": response, "context": chat_sessions[user_id][-5:]})  # Return last 5 messages

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
