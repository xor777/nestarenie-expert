import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("Не найден ключ OPENAI_API_KEY в переменных окружения")
    exit(1)

client_openai = OpenAI(api_key=OPENAI_API_KEY)

TEMPERATURE = float(os.getenv('TEMPERATURE', 0.1))
MIN_RELEVANCE = float(os.getenv('MIN_RELEVANCE', 0.9))
MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', 1000))
DIRECT_ANSWER_RELEVANCE = float(os.getenv('DIRECT_ANSWER_RELEVANCE', 0.98))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    print("Не найден токен TELEGRAM_TOKEN в переменных окружения")
    exit(1)

if 'EMBEDDING_MODEL' in os.environ:
    del os.environ['EMBEDDING_MODEL']
load_dotenv()
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("questions")

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        text = " ".join(text.split())
        
        if len(text) > MAX_INPUT_TOKENS * 4:
            print("Предупреждение: текст слишком длинный, будет использована только его часть")
            text = text[:MAX_INPUT_TOKENS * 4]
            
        response = client_openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Ошибка при получении эмбеддинга: {str(e)}")
        return None

def save_generated_answer(question: str, answer: str, reference: str = "GPT") -> None:
    try:
        embedding = get_embedding(question)
        if embedding:
            collection.add(
                embeddings=[embedding],
                documents=[question],
                metadatas=[{"answer": answer, "reference": reference, "is_generated": True}],
                ids=[str(collection.count() + 1)]
            )
    except Exception as e:
        print(f"Ошибка при сохранении ответа: {str(e)}")

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
            print(f"relevance: {relevance} | skipped: {question}")
            continue
        print(f"relevance: {relevance} | added: {question}")
        context.append({
            "question": question,
            "answer": metadata["answer"],
            "reference": metadata["reference"],
            "relevance": relevance,
            "is_generated": metadata.get("is_generated", False)
        })
    return context

def generate_response(query: str, context: List[Dict]) -> str:
    if not context:
        return "Извините, в базе знаний нет достаточно релевантной информации."
    
    context_text = "\n\n".join([
        f"ФРАГМЕНТ #{i+1}\nКОНТЕКСТ:\n{c['answer']}\nURL:\n{c['reference']}"
        for i, c in enumerate(sorted(context, key=lambda x: x['relevance'], reverse=True))
    ])

    system_message = '''Ты - специализированная медицинская система для точных ответов на вопросы о биотехнологиях и науке о старении.

        ТВОЯ ЗАДАЧА:
        Дать исчерпывающий ответ на вопрос, используя ВСЮ релевантную информацию из предоставленных фрагментов.

        СТРОГИЙ ФОРМАТ ОТВЕТА:
        1. Подробный ответ на вопрос (без заголовков), включающий ВСЕ важные детали из релевантных фрагментов
        2. Пустая строка
        3. "Ссылки:" и список использованных URL, по одному в строке

        ПРАВИЛА СОСТАВЛЕНИЯ ОТВЕТА:
        1. Включай ВСЮ существенную информацию из фрагментов, которая относится к вопросу
        2. Используй все важные детали, цифры, факты и пояснения из релевантных фрагментов
        3. Не пропускай значимые подробности, если они помогают полнее ответить на вопрос
        4. Ответ должен быть максимально информативным в рамках заданного вопроса
        5. Структурируй информацию логично и последовательно

        ПРАВИЛА РАБОТЫ С ФРАГМЕНТАМИ И ССЫЛКАМИ:
        1. Каждый фрагмент имеет номер, контекст и URL
        2. В разделе "Ссылки:" указывай только УНИКАЛЬНЫЕ URL из использованных фрагментов
        3. Если один и тот же URL встречается в разных фрагментах, укажи его только ОДИН раз
        4. Копируй URL в точности как они указаны, без добавления номеров, маркеров списка или других символов
        5. Если информация из фрагмента не использована в ответе - НЕ включай его URL
        6. URL должны идти каждый с новой строки
        7. Не добавляй никакого форматирования к URL'''

    user_message = f'''КОНТЕКСТ:
        {context_text}

        ВОПРОС:
        {query}

        ТРЕБОВАНИЯ:
        1. Дай ПОДРОБНЫЙ ответ на вопрос, включая ВСЕ важные детали из релевантных фрагментов
        2. Не пропускай существенную информацию, которая помогает лучше ответить на вопрос
        3. В разделе "Ссылки:" должны быть ТОЛЬКО URL из тех фрагментов, информация из которых использована в ответе
        4. URL должны быть скопированы точно, без изменений и дополнительных символов'''
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Ошибка при генерации ответа: {str(e)}")
        return "Извините, произошла техническая ошибка. Пожалуйста, попробуйте переформулировать вопрос."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Здравствуйте! Я - медицинская экспертная система в области биотехнологий и науки о старении. "
        "Задавайте ваши вопросы, и я постараюсь ответить на них, основываясь на научных данных."
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
    
    # Если контекст пустой, значит вопрос не по теме
    if not relevant_context:
        await update.message.reply_text("Извините, в базе знаний нет релевантной информации по вашему вопросу.")
        return
    
    # Если есть ответ с высокой релевантностью - возвращаем его
    if relevant_context[0]['relevance'] >= DIRECT_ANSWER_RELEVANCE:
        most_relevant = relevant_context[0]
        response = f"📖 {most_relevant['answer']}\n\nИсточник: {most_relevant['reference']}"
    # Если есть контекст, но нет ответа с высокой релевантностью - генерируем
    else:
        generated_response = generate_response(query, relevant_context)
        save_generated_answer(query, generated_response)
        response = f"🧠 {generated_response}"
    
    if len(response) > 4096:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i+4096])
    else:
        await update.message.reply_text(response)

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