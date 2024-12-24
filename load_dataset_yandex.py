import chromadb
import shutil
import os
import pandas as pd
from tqdm import tqdm
import requests
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv('FOLDER_ID')
IAM_TOKEN = os.getenv('YC_IAM_TOKEN')

if not FOLDER_ID:
    print("Error: FOLDER_ID not found in environment variables")
    exit(1)
    
if not IAM_TOKEN:
    print("Error: YC_IAM_TOKEN not found in environment variables")
    exit(1)

def get_embedding(text: str) -> list:
    try:
        request_body = {
            "modelUri": f"emb://{FOLDER_ID}/text-search-query",
            "text": text
        }
        
        headers = {
            "Authorization": f"Bearer {IAM_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
            headers=headers,
            json=request_body
        )
        
        if response.status_code == 200:
            return response.json()['embedding']
        else:
            print(f"Error getting embedding: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Yandex API request error: {str(e)}")
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
    load_dataset()