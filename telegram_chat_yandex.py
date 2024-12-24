import os
import chromadb
import requests
from dotenv import load_dotenv
from typing import Optional, List, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

FOLDER_ID = os.getenv('FOLDER_ID')
IAM_TOKEN = os.getenv('YC_IAM_TOKEN')
GENERATION_MODEL = os.getenv('GENERATION_MODEL', 'yandexgpt')

if not FOLDER_ID:
    print("Не найден FOLDER_ID в переменных окружения")
    exit(1)
    
if not IAM_TOKEN:
    print("Не найден YC_IAM_TOKEN в переменных окружения")
    exit(1)

class BotConfig:
    def __init__(self):
        self.temperature = float(os.getenv('TEMPERATURE', 0.3))
        self.min_relevance = float(os.getenv('MIN_RELEVANCE', 0.7))
        self.max_tokens = int(os.getenv('MAX_TOKENS', 8000))

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    print("Не найден токен TELEGRAM_TOKEN в переменных окружения")
    exit(1)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("questions")

def get_embedding(text: str, config: BotConfig) -> Optional[List[float]]:
    try:
        text = " ".join(text.split())
        
        if len(text) > config.max_tokens * 4:
            print("Предупреждение: текст слишком длинный, будет использована только его часть")
            text = text[:config.max_tokens * 4]
        
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
            print(f"Ошибка при получении эмбеддинга: {response.status_code}")
            print(f"Ответ: {response.text}")
            return None
            
    except Exception as e:
        print(f"Ошибка API Yandex: {str(e)}")
        return None

def get_relevant_context(query: str, config: BotConfig) -> List[Dict]:
    query_embedding = get_embedding(query, config)
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
        if relevance < config.min_relevance:
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

def generate_response(query: str, context: List[Dict], config: BotConfig) -> str:
    if not context:
        return "Извините, в базе знаний нет достаточно релевантной информации..."
    
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
        7. Не добавляй никакого форматирования к URL

        ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:
        [Подробный, детальный ответ, включающий все важные факты, цифры и пояснения из релевантных фрагментов, которые помогают полно ответить на вопрос]

        Ссылки:
        https://example1.com
        https://example2.com

        ЗАПРЕЩЕНО:
        1. Пропускать важные детали из релевантных фрагментов
        2. Чрезмерно сокращать информацию
        3. Добавлять любые заголовки или дополнительные разделы
        4. Включать информацию, не относящуюся к вопросу
        5. Добавлять нумерацию или маркеры к URL
        6. Добавлять текст или описание рядом с URL
        7. Включать URL из фрагментов, информация которых не использована
        8. Изменять или форматировать URL
        9. Добавлять пробелы или символы в начале строк с URL'''

    user_message = f'''КОНТЕКСТ:
        {context_text}

        ВОПРОС:
        {query}

        ТРЕБОВАНИЯ:
        1. Дай ПОДРОБНЫЙ ответ на вопрос, включая ВСЕ важные детали из релевантных фрагментов
        2. Не пропускай существенную информацию, которая помогает лучше ответить на вопрос
        3. В разделе "Ссылки:" должны быть ТОЛЬКО URL из тех фрагментов, информация из которых использована в ответе
        4. URL должны быть скопированы точно, без изменений и дополнительных символов'''

    #print(context_text)
    
    try:
        request_body = {
            "modelUri": f"gpt://{FOLDER_ID}/{GENERATION_MODEL}",
            "completionOptions": {
                "stream": False,
                "temperature": config.temperature,
                "maxTokens": "2000"
            },
            "messages": [
                {"role": "system", "text": system_message},
                {"role": "user", "text": user_message}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {IAM_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=request_body
        )
        
        if response.status_code == 200:
            return response.json()['result']['alternatives'][0]['message']['text']
        else:
            print(f"Ошибка при генерации ответа: {response.status_code}")
            print(f"Ответ: {response.text}")
            return "Извините, произошла техническая ошибка. Пожалуйста, попробуйте переформулировать вопрос."
    
    except Exception as e:
        print(f"Ошибка при генерации ответа: {str(e)}")
        return "Извините, произошла техническая ошибка. Пожалуйста, попробуйте переформулировать вопрос."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Здравствуйте! Я - медицинская экспертная система в области биотехнологий и науки о старении. "
        "Задавайте ваши вопросы, и я постараюсь ответить на них, основываясь на научных данных."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_config = context.bot_data.get('config')
    if not bot_config:
        return

    await update.message.reply_text(
        "Я могу ответить на ваши вопросы о биотехнологиях и науке о старении. "
        "Просто напишите свой вопрос, и я постараюсь найти релевантную информацию в своей базе знаний.\n\n"
        "Доступные команды:\n"
        "/start - Начать общение\n"
        "/help - Показать это сообщение\n"
        "/temperature <число> - Изменить температуру генерации\n\n"
        f"Текущая температура: {bot_config.temperature}\n"
        "  0 = консервативные, точные ответы\n"
        "  1 = креативные, разнообразные ответы"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_config = context.bot_data.get('config')
    if not bot_config:
        return

    query = update.message.text
    user = update.effective_user
    
    print(f"\n{'='*20} вопрос: {'='*20}")
    print(f"Username: {user.username or user.id} | Name: {user.first_name}")
    print(f"Question: {query}")
    print(f"{'='*60}")
    
    await update.message.chat.send_action(action="typing")
    
    relevant_context = get_relevant_context(query, bot_config)

    '''    
    if relevant_context and relevant_context[0]['relevance'] > 0.7:
        most_relevant = relevant_context[0]
        await update.message.reply_text(
            f"💡 Наиболее релевантный ответ из базы знаний "
            f"(релевантность: {most_relevant['relevance']:.2f}):\n\n"
            f"{most_relevant['answer']}\n\n"
            f"Источник: {most_relevant['reference']}"
        )
    else:
        await update.message.reply_text(
            "❌ В базе знаний не найдено достаточно релевантных ответов на ваш вопрос."
        )
    '''
    await update.message.chat.send_action(action="typing")
    response = generate_response(query, relevant_context, bot_config)
    
    if len(response) > 4096:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i+4096])
    else:
        await update.message.reply_text(response)

async def set_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_config = context.bot_data.get('config')
    if not bot_config:
        return

    if not context.args:
        await update.message.reply_text(
            "Пожалуйста, укажите значение температуры от 0 до 1.\n"
            "Пример: /temperature 0.7\n"
            f"Текущее значение: {bot_config.temperature}"
        )
        return

    try:
        new_temp = float(context.args[0])
        if not 0 <= new_temp <= 1:
            await update.message.reply_text(
                "Значение температуры должно быть в диапазоне от 0 до 1.\n"
                f"Текущее значение: {bot_config.temperature}"
            )
            return
        
        bot_config.temperature = new_temp
        await update.message.reply_text(
            f"Температура успешно изменена на {bot_config.temperature}.\n\n"
            "Чем ближе к 0, тем более консервативные и предсказуемые ответы.\n"
            "Чем ближе к 1, тем более креативные и разнообразные ответы."
        )
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, укажите корректное числовое значение.\n"
            "Пример: /temperature 0.7\n"
            f"Текущее значение: {bot_config.temperature}"
        )

def main() -> None:
    if not os.path.exists("./chroma_db"):
        print("База данных не найдена. Сначала запустите load_dataset.py")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.bot_data['config'] = BotConfig()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("temperature", set_temperature))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main() 