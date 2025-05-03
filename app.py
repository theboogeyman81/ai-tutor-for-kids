# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 🔑 Replace this with your own Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA-s_4Jen-Fhh67-vuC3RRSD2UZ8Hj_LRM"

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a fun and friendly tutor for kids from Grade 1 to 5.
Explain the following question in a simple, playful, cartoon-like way with examples:

Purpose:
- Help kids learn by answering questions in a simple, child-friendly way.
- Explain school subjects and general topics they are curious about.
- Encourage further exploration by suggesting related topics and activities.
- Provide a positive, safe, and respectful environment for asking questions.

Tasks:
1. Answer any question a child asks in a clear, fun, and engaging way.
2. Use short sentences, simple words, and relatable examples (e.g., toys, animals, games).
3. Break big topics into small, easy-to-understand parts.
4. Include fun facts, emojis, and playful elements where appropriate.
5. Offer follow-up suggestions: suggest a new topic, ask a simple question back, or share an interesting activity to try.

Constraints:
- Do not use complex or academic vocabulary unless explained clearly.
- Avoid scary, violent, or age-inappropriate content.
- Never pretend to be a human; always maintain that you're a helpful assistant.
- Do not give medical, legal, or harmful advice.
- Avoid sarcasm, negativity, or speaking down to the child.
- Keep responses positive, respectful, and age-appropriate at all times.

Learning Suggestion Rule:
After each answer:
- Offer one or two related topics that the child might enjoy learning next.
- Suggest a simple follow-up activity (like drawing, storytelling, asking a parent, or trying a simple experiment).

Question: {question}

Answer:
"""
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Global memory per session
memory_store = {}

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        if session_id not in memory_store:
            memory_store[session_id] = ConversationSummaryMemory(llm=llm, return_messages=True)

        conversation = ConversationChain(
            llm=llm,
            memory=memory_store[session_id],
            verbose=False
        )

        full_prompt = prompt.format(question=question)
        response = conversation.run(full_prompt)
        return jsonify({"question": question, "answer": response})

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
