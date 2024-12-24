import os
import chromadb
import requests
from dotenv import load_dotenv
from typing import Optional, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

FOLDER_ID = os.getenv('FOLDER_ID')
IAM_TOKEN = os.getenv('YC_IAM_TOKEN')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

if not all([FOLDER_ID, IAM_TOKEN, TELEGRAM_TOKEN]):
    print("Ошибка: проверьте наличие всех необходимых токенов в .env файле")
    exit(1)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("questions")

def get_embedding(text: str) -> Optional[list]:
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
        
        return response.json()['embedding'] if response.status_code == 200 else None
            
    except Exception as e:
        print(f"Ошибка API: {str(e)}")
        return None

def get_most_relevant_answer(query: str) -> Optional[Dict]:
    query_embedding = get_embedding(query)
    if not query_embedding:
        return None
        
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return None
            
        relevance = 1 - results['distances'][0][0]
        if relevance < 0.7:
            return None
            
        print(f"relevance: {relevance} | used: {results['documents'][0][0]}")
        return {
            "answer": results['metadatas'][0][0]["answer"],
            "reference": results['metadatas'][0][0]["reference"],
            "relevance": relevance
        }
        
    except Exception as e:
        print(f"Ошибка базы данных: {str(e)}")
        return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Здравствуйте! Задавайте вопросы, и я найду релевантные ответы в базе знаний."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    user = update.effective_user
    
    print(f"\n{'='*20} вопрос: {'='*20}")
    print(f"Username: {user.username or user.id} | Name: {user.first_name}")
    print(f"Question: {query}")
    print(f"{'='*60}")
    
    await update.message.chat.send_action(action="typing")
    
    answer = get_most_relevant_answer(query)
    if answer:
        await update.message.reply_text(
            f"💡 Наиболее релевантный ответ из базы знаний "
            f"(релевантность: {answer['relevance']:.2f}):\n\n"
            f"{answer['answer']}\n\n"
            f"Источник: {answer['reference']}"
        )
    else:
        await update.message.reply_text(
            "❌ В базе знаний не найдено достаточно релевантных ответов на ваш вопрос."
        )

def main() -> None:
    if not os.path.exists("./chroma_db"):
        print("База данных не найдена. Сначала запустите load_dataset.py")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main() 