from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pinecone
from groq import Groq
import pickle

app = Flask(__name__)

# Initialize the model, Pinecone, and Groq client
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = "preg"
pc = pinecone.Pinecone(api_key="fd50e191-da01-45d7-9d56-515b3cf3b8db")
client = Groq(
    api_key="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",
)

def get_context(ques, tot2):
    index = pc.Index(index_name)
    ques_emb = model.encode(ques)
    DB_response = index.query(
        vector=ques_emb.tolist(),
        top_k=3,
        include_values=True
    )

    if DB_response is None or 'matches' not in DB_response:
        return ""

    cont = ""
    for i in range(len(DB_response['matches'])):
        try:
            chunk_index = int(DB_response['matches'][i]['id'][3:]) - 1
            cont += tot2[chunk_index]
        except (IndexError, ValueError) as e:
            pass
    return cont

def load_chunks():
    with open('tot_chunks.pkl', 'rb') as f:
        return pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question")
    if query:
        tot_chunks = load_chunks()
        context = get_context(query, tot_chunks)
        if context:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Context: {context}, Analyse and understand the above context completely , if the query is not relevent to the context , then reply in general, Query: {query}",
                    }
                ],
                model="llama3-70b-8192",
            )
            response_text = chat_completion.choices[0].message.content
            return jsonify({"response": response_text})
        else:
            return jsonify({"response": "No relevant context found for the given question."})
    return jsonify({"response": "No question provided."})

if __name__ == "__main__":
    app.run(debug=True)
