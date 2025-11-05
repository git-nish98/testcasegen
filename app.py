from flask import Flask, render_template, request, jsonify
from gpt4all import GPT4All
import json, os

app = Flask(__name__)


MODEL_PATH = "Llama-3.2-3B-Instruct-Q4_0.gguf"
model = GPT4All(MODEL_PATH)


HISTORY_FILE = "chat_history.json"

def load_history():
    """Load previous chat history from file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    """Save updated chat history."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def get_relevant_context(prompt, history, limit=6):
    """
    Select only contextually relevant past messages based on word overlap.
    """
    relevant_context = []
    prompt_words = set(prompt.lower().split())

    for msg in reversed(history):
        text_words = set(msg["text"].lower().split())
        if len(prompt_words.intersection(text_words)) > 1:
            relevant_context.append(msg)
        if len(relevant_context) >= limit:
            break

    relevant_context.reverse()
    return relevant_context

def ask_model(prompt, history):
    """
    Send prompt to GPT4All with relevant history and handle long responses.
    """
    relevant_history = get_relevant_context(prompt, history)
    context = "\n".join(
        f"{'User' if msg['sender'] == 'You' else 'Assistant'}: {msg['text']}"
        for msg in relevant_history
    )

    full_prompt = (
        "You are a helpful and accurate AI assistant. "
        "Respond in a professional and concise manner, but provide full details when needed.\n\n"
        f"Previous relevant conversation:\n{context}\n\n"
        f"User: {prompt}\nAssistant:"
    )

    final_response = ""
    with model.chat_session() as session:
        for token in session.generate(
            full_prompt,
            temp=0.7,
            max_tokens=800,  
            streaming=True    
        ):
            final_response += token

    for stop_word in ["User:", "Assistant:"]:
        if stop_word in final_response:
            final_response = final_response.split(stop_word)[0].strip()
    print(final_response)
    return final_response.strip()


@app.route("/")
def index():
    history = load_history()
    return render_template("chat.html", chat=history)

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    history = load_history()

    history.append({"sender": "You", "text": user_input})


    response = ask_model(user_input, history)


    history.append({"sender": "Bot", "text": response})


    save_history(history)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
