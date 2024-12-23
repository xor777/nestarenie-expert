import os
import chromadb
from dotenv import load_dotenv
from typing import Optional, List, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests

load_dotenv()

TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
MIN_RELEVANCE = float(os.getenv('MIN_RELEVANCE', 0.7))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 8000))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'evilfreelancer/enbeddrus')
GENERATION_MODEL = os.getenv('GENERATION_MODEL', 'llama3.2')

if not TELEGRAM_TOKEN:
    print("Error: TELEGRAM_TOKEN not found in environment variables")
    exit(1)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("questions")

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        text = " ".join(text.split())
        
        if len(text) > MAX_TOKENS * 4:  
            print("Предупреждение: текст слишком длинный, будет обрезан")
            text = text[:MAX_TOKENS * 4]
        
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
            print(f"Ошибка получения эмбеддинга: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка запроса к Ollama: {str(e)}")
        return None

def get_relevant_context(query: str) -> List[Dict]:
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
        
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Database search error: {str(e)}")
        return []
    
    context = []
    for question, metadata, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        relevance = 1 - distance
        if relevance < MIN_RELEVANCE: 
            print(f"relevance: {relevance} | skipped: {question}")
            continue
        print(f"relevance: {relevance} | added: {question}")
        context.append({
            "question": question,
            "answer": metadata["answer"],
            "reference": metadata["reference"],
            "relevance": relevance
        })
    return context

def generate_response(query: str, context: List[Dict]) -> str:
    if not context:
        return "Извините, в моей базе знаний нет достаточно релевантной информации для ответа на ваш вопрос. Пожалуйста, попробуйте переформулировать вопрос."
    
    context_text = "\n\n".join([
        f"[ДАННЫЕ]\n{c['answer']}\n[ИСТОЧНИКИ]\n{c['reference']}"
        for c in sorted(context, key=lambda x: x['relevance'], reverse=True)
    ])

    system_prompt = '''Ты - специализированная медицинская система для точных ответов на вопросы о биотехнологиях и науке о старении.

        ТВОЯ ЗАДАЧА:
        Дать точный ответ на вопрос пользователя, основываясь СТРОГО на научных данных из предоставленного контекста.
        ВСЕГДА отвечай на русском языке.

        СТРОГИЙ ФОРМАТ ОТВЕТА:
        1. Всегда отвечай на русском языке
        2. Один параграф текста с ответом на вопрос
        3. Если есть источники, пиши "Ссылки:" на новой строке и список источников
        4. Если источников нет, заканчивай ответ параграфом текста

        КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
        1. ИСПОЛЬЗУЙ только информацию из раздела [ДАННЫЕ]
        2. ОТВЕЧАЙ прямо на поставленный вопрос
        3. ПРИДЕРЖИВАЙСЯ заданного формата ответа
        4. ПРИЗНАВАЙ отсутствие информации, если её нет
        5. НЕ ПИШИ ничего лишнего
        6. ВСЕГДА отвечай на русском языке

        ТЫ БУДЕШЬ ОШТРАФОВАН ЗА:
        - Отклонение от заданного формата ответа
        - Использование информации не из раздела [ДАННЫЕ]
        - Добавление лишней информации
        - Интерпретацию данных
        - Предположения и выводы
        - Ответ не на русском языке'''

    prompt = f'''КОНТЕКСТ:
        {context_text}

        ВОПРОС ПОЛЬЗОВАТЕЛЯ:
        {query}

        СТРОГИЕ ТРЕБОВАНИЯ К ОТВЕТУ:
        1. Один параграф текста с ответом на русском языке
        2. После ответа на новой строке напиши "Источники:" и источники из [ИСТОЧНИКИ], ТОЛЬКО если они есть
        3. Никакой лишней информации
        4. Если данных нет или их недостаточно - так и напиши в одном параграфе'''

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': GENERATION_MODEL,
                'prompt': prompt,
                'temperature': TEMPERATURE,
                'stream': False
            }
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            print(f"Ошибка генерации ответа: {response.status_code}")
            return "Произошла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."
    except Exception as e:
        print(f"Ошибка запроса к Ollama: {str(e)}")
        return "Произошла техническая ошибка. Пожалуйста, попробуйте еще раз."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Здравствуйте! Я - медицинская экспертная система в области биотехнологий и науки о старении. "
        "Задавайте ваши вопросы, и я постараюсь ответить на них, основываясь на научных данных."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я могу ответить на ваши вопросы о биотехнологиях и науке о старении. "
        "Просто напишите свой вопрос, и я постараюсь найти релевантную информацию в своей базе знаний."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    user = update.effective_user
    
    print(f"\n{'='*20} вопрос: {'='*20}")
    print(f"Username: {user.username or user.id} | Name: {user.first_name}")
    print(f"Question: {query}")
    
    await update.message.chat.send_action(action="typing")
    
    relevant_context = get_relevant_context(query)
    response = generate_response(query, relevant_context)
    
    if len(response) > 4096:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i+4096])
    else:
        await update.message.reply_text(response)

def main() -> None:
    if not os.path.exists("./chroma_db"):
        print("Error: Database not found. Run load_dataset.py first")
        return

    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            print("Error: Ollama server is not available")
            exit(1)
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        print("Make sure Ollama is running and available at localhost:11434")
        exit(1)

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running")
    application.run_polling()

if __name__ == "__main__":
    main() 