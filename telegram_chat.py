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
    print("Не найден ключ OPENAI_API_KEY в переменных окружения, добавьте его в .env файл")
    exit(1)

client_openai = OpenAI(api_key=OPENAI_API_KEY)

TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
MIN_RELEVANCE = float(os.getenv('MIN_RELEVANCE', 0.7))
MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', 1000))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    print("Не найден токен TELEGRAM_TOKEN в переменных окружения, добавьте его в .env файл")
    exit(1)

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
            "relevance": relevance
        })
    return context

def generate_response(query: str, context: List[Dict]) -> str:
    if not context:
        return "Извините, в базе знаний нет достаточно релевантной информации для ответа на ваш вопрос. Пожалуйста, переформулируйте вопрос или задайте другой вопрос о биотехнологиях и науке о старении."
    
    context_text = "\n\n".join([
        f"[ДАННЫЕ]\n{c['answer']}\n[ИСТОЧНИКИ]\n{c['reference']}"
        for c in sorted(context, key=lambda x: x['relevance'], reverse=True)
    ])

    system_prompt = '''Ты - специализированная медицинская система для точных ответов на вопросы о биотехнологиях и науке о старении.

        ТВОЯ ЗАДАЧА:
        Дать точный ответ на вопрос пользователя, основываясь СТРОГО на научных данных из предоставленного контекста.

        СТРОГИЙ ФОРМАТ ОТВЕТА:
        1. Один параграф текста с ответом на вопрос
        2. Если есть источники, пиши "Ссылки:" на новой строке и список источников
        3. Если источников нет, заканчивай ответ параграфом текста

        КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
        1. ИСПОЛЬЗУЙ только информацию из раздела [ДАННЫЕ]
        2. ОТВЕЧАЙ прямо на поставленный вопрос
        3. ПРИДЕРЖИВАЙСЯ заданного формата ответа
        4. ПРИЗНАВАЙ отсутствие информации, если её нет
        5. НЕ ПИШИ ничего лишнего

        ТЫ БУДЕШЬ ОШТРАФОВАН ЗА:
        - Отклонение от заданного формата ответа
        - Использование информации не из раздела [ДАННЫЕ]
        - Добавление лишней информации
        - Интерпретацию данных
        - Предположения и выводы'''

    user_prompt = f'''КОНТЕКСТ:
        {context_text}

        ВОПРОС ПОЛЬЗОВАТЕЛЯ:
        {query}

        СТРОГИЕ ТРЕБОВАНИЯ К ОТВЕТУ:
        1. Один параграф текста с ответом
        2. После ответа на новой строке напиши "Ссылки:" и источники из [ИСТОЧНИКИ], ТОЛЬКО если они есть
        3. Никакой лишней информации
        4. Если данных нет или их недостаточно - так и напиши в одном параграфе'''

    #print(f"Context: {context_text}")
    #print(f"System prompt: {system_prompt}")
    #print(f"User prompt: {user_prompt}")
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=2000
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

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main() 