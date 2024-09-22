import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer



def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(query, index, chunks, embedder, top_k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=True).detach().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]


def generate_response(question, context):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=512, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def chat_with_pdf(pdf_path, user_query):
    # Extract and chunk the text from PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Embed chunks and create FAISS index
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    # Search for relevant chunks
    relevant_chunks = search_index(user_query, index, chunks, embedder)
    context = " ".join(relevant_chunks)

    # Generate response using Flan-T5
    response = generate_response(user_query, context)
    return response


if __name__ == "__main__":
    pdf_path = 'OSINT_Handbook_June-2018_Final.pdf'
    user_query = "What is the main idea of the first chapter?"

    response = chat_with_pdf(pdf_path, user_query)
    print("AI Response:", response)
