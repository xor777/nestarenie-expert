import chromadb
import shutil
import os
import pandas as pd
from tqdm import tqdm
import requests
import json
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'evilfreelancer/enbeddrus')

def get_embedding(text: str) -> list:
    try:
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={
                'model': EMBEDDING_MODEL,
                'prompt': text
            }
        )
        if response.status_code == 200:
            return response.json()['embedding']
        else:
            print(f"Error getting embedding: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ollama request error: {str(e)}")
        return None

def load_dataset():
    if os.path.exists("./chroma_db"):
        print("Removing existing database...")
        shutil.rmtree("./chroma_db")
    
    client_chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = client_chroma.create_collection(
        name="questions",
        metadata={"hnsw:space": "cosine"}
    )

    print("Loading dataset...")
    df = pd.read_csv('dataset.csv')
    
    questions = df['Вопрос'].tolist()
    answers = df['Ответ'].tolist()
    references = df['Ссылка'].tolist()
    ids = [str(i) for i in range(len(questions))]
    
    print("Generating embeddings...")
    embeddings = []
    for q in tqdm(questions, desc="Processing"):
        emb = get_embedding(q)
        if emb is None:
            print(f"Skipping question due to error: {q}")
            continue
        embeddings.append(emb)
    
    if len(embeddings) != len(questions):
        print("Warning: not all embeddings were successfully generated")
        valid_indices = list(range(len(embeddings)))
        questions = [questions[i] for i in valid_indices]
        answers = [answers[i] for i in valid_indices]
        references = [references[i] for i in valid_indices]
        ids = [str(i) for i in range(len(embeddings))]
    
    print("Adding data to ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=questions,
        metadatas=[{"answer": a, "reference": r} for a, r in zip(answers, references)],
        ids=ids
    )
    
    print("Database created and populated successfully")

if __name__ == "__main__":
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            print("Error: Ollama server is not available")
            exit(1)
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        print("Make sure Ollama is running and available at localhost:11434")
        exit(1)
        
    load_dataset() 