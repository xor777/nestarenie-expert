import chromadb
import shutil
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

if 'EMBEDDING_MODEL' in os.environ:
    del os.environ['EMBEDDING_MODEL']
load_dotenv()

client_openai = OpenAI()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

def get_embedding(text):
    response = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

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
    embeddings = [get_embedding(q) for q in tqdm(questions, desc="Processing")]
    
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