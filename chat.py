import torch
import json
import random
import nltk
from nltk.stem import PorterStemmer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load intents.json
file_path = "intents.json"
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE = "data.pth"
data = torch.load(FILE)

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

conversation_history = []

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

def chat():
    print("MindMate is ready to chat! (type 'quit' to exit)")

    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        conversation_history.append(f"User: {sentence}")

        # Process input with the trained model
        tokenized_sentence = tokenize(sentence)
        X = bag_of_words(tokenized_sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        prob = torch.softmax(output, dim=1)[0][predicted.item()]

        if prob.item() >= 0.75:
            tag = tags[predicted.item()]
        else:
            tag = get_best_matching_intent(sentence)

        response = random.choice(fallback_responses)
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                break

        conversation_history.append(f"{bot_name}: {response}")

        print(f"{bot_name}: {response}")
        print("\nChat History:")
        for message in conversation_history[-5:]:  # Show last 5 exchanges
            print(message)

if __name__ == "__main__":
    chat()
