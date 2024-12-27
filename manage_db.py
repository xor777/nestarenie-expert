import chromadb
import argparse
from typing import Tuple

def get_stats() -> Tuple[int, int]:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("questions")
    
    results = collection.get(
        include=["metadatas"]
    )
    
    total_count = len(results['metadatas'])
    generated_count = sum(1 for meta in results['metadatas'] if meta.get('is_generated', False))
    
    return total_count, generated_count

def delete_generated() -> int:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("questions")
    
    results = collection.get(
        include=["metadatas", "documents", "embeddings"]
    )
    
    keep_indices = [
        i for i, meta in enumerate(results['metadatas'])
        if not meta.get('is_generated', False)
    ]
    
    if not keep_indices:
        print("Нет записей для сохранения")
        return 0
    
    client.delete_collection("questions")
    new_collection = client.create_collection(
        name="questions",
        metadata={"hnsw:space": "cosine"}
    )
    
    new_collection.add(
        ids=[str(i) for i in range(len(keep_indices))],
        embeddings=[results['embeddings'][i] for i in keep_indices],
        documents=[results['documents'][i] for i in keep_indices],
        metadatas=[results['metadatas'][i] for i in keep_indices]
    )
    
    deleted_count = len(results['embeddings']) - len(keep_indices)
    return deleted_count

def main():
    parser = argparse.ArgumentParser(description='Утилита для управления базой данных ChromaDB')
    parser.add_argument('--stats', action='store_true', help='Показать статистику базы данных')
    parser.add_argument('--delete-generated', action='store_true', help='Удалить все сгенерированные записи')
    
    args = parser.parse_args()
    
    if not args.stats and not args.delete_generated:
        parser.print_help()
        return
    
    try:
        if args.stats:
            total, generated = get_stats()
            print(f"Всего записей: {total}")
            print(f"Сгенерированных записей: {generated}")
            print(f"Оригинальных записей: {total - generated}")
        
        if args.delete_generated:
            deleted = delete_generated()
            print(f"Удалено {deleted} сгенерированных записей")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        print("Убедитесь, что база данных существует и доступна")

if __name__ == "__main__":
    main() 