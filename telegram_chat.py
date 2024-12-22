import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()
client_openai = OpenAI()

TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
MIN_RELEVANCE = float(os.getenv('MIN_RELEVANCE', 0.7))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 8000))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("questions")

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        text = " ".join(text.split())
        
        if len(text) > MAX_TOKENS * 4:  
            print("Предупреждение: текст слишком длинный, будет использована только его часть")
            text = text[:MAX_TOKENS * 4]
        
        response = client_openai.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Ошибка при получении эмбеддинга: {str(e)}")
        return None

def get_relevant_context(query: str) -> List[Dict]:
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
        
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Ошибка при поиске в базе данных: {str(e)}")
        return []
    
    context = []
    for question, metadata, distance in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        relevance = 1 - distance
        if relevance < MIN_RELEVANCE: 
            continue
        context.append({
            "question": question,
            "answer": metadata["answer"],
            "reference": metadata["reference"],
            "relevance": relevance
        })
    return context

def generate_response(query: str, context: List[Dict]) -> str:
    if not context:
        return "Извините, но в моей базе знаний нет достаточно релевантной информации для ответа на ваш вопрос. Пожалуйста, переформулируйте вопрос или задайте другой вопрос о биотехнологиях и науке о старении."
    
    system_prompt = """Ты - специализированная медицинская экспертная система в области биотехнологий и науки о старении. 
    
    КЛЮЧЕВЫЕ ПРИНЦИПЫ РАБОТЫ:
    1. Безопасность превыше всего - никогда не выдумывай информацию и не делай предположений
    2. Используй ТОЛЬКО информацию из предоставленного контекста
    3. Не упускай детали из контекста, все детали важны
    4. Если информации недостаточно или есть неуверенность - честно признай это
    5. Указывай источники информации, если они предоставлены в контексте
    
    ФОРМАТ ОТВЕТА:
    1. Сначала дай прямой ответ на вопрос пользователя
    2. Затем предоставь дополнительную информацию и объяснения, если они есть в контексте, если нет - ничего не пиши
    3. В конце обязательно укажи ссылки, которые тебе предоставлены в контексте, если нет - не добавляй
    
    ВАЖНЫЕ ПРАВИЛА:
    - Если в контексте нет прямого ответа на вопрос - явно сообщи об этом
    - Не экстраполируй данные исследований на другие случаи
    - Не давай медицинских рекомендаций
    - Если пользователь спрашивает о чем-то, что выходит за рамки предоставленных данных, объясни, что ты не можешь ответить на вопрос
    
    ЗАПРЕЩЕНО:
    - Додумывать или предполагать информацию
    - Обобщать результаты исследований
    - Давать медицинские советы
    - Интерпретировать данные шире, чем они представлены в источниках"""

    context_text = "\n\n".join([
        f"Релевантный фрагмент (релевантность {c['relevance']:.2%}):\n"
        #f"Вопрос: {c['question']}\n"
        f"Ответ: {c['answer']}\n"
        f"Ссылки: {c['reference']}"
        for c in context
    ])
    
    user_prompt = f"""Контекст:
{context_text}

Вопрос пользователя: {query}

Ответь на вопрос, строго придерживаясь предоставленной информации. 
Если информации недостаточно или есть какие-то неясности - обязательно укажи это."""

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE
    )
    
    return response.choices[0].message.content

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
    print(f"{'='*60}")
    
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
        print("База данных не найдена. Сначала запустите load_dataset.py")
        return
        
    if not TELEGRAM_TOKEN:
        print("Не найден токен Telegram бота. Добавьте TELEGRAM_TOKEN в .env файл")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main() 