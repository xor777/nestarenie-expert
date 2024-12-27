import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import json

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

def get_relevant_context(query: str, query_embedding: List[float], include_generated: bool = True) -> List[Dict]:
    print(f"searching... [pre-generated {'included' if include_generated else 'excluded'}]")
    
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
        is_generated = metadata.get('is_generated', False)
        
        # Пропускаем сгенерированные ответы, если include_generated=False
        if not include_generated and is_generated:
            print(f"relevance: {relevance} | skipped (generated): {question}")
            continue
            
        if relevance < MIN_RELEVANCE:
            print(f"relevance: {relevance} | skipped (relevance): {question}")
            continue
            
        print(f"relevance: {relevance} | added: {question}")
        context.append({
            "question": question,
            "answer": metadata["answer"],
            "reference": metadata["reference"],
            "relevance": relevance,
            "is_generated": is_generated
        })
    return context

def generate_response(query: str, context: List[Dict]) -> Dict:
    if not context:
        return None
    
    context_text = "\n\n".join([
        f"ФРАГМЕНТ #{i+1}\nВОПРОС:\n{c['question']}\nОТВЕТ:\n{c['answer']}\nURL:\n{c['reference']}"
        for i, c in enumerate(sorted(context, key=lambda x: x['relevance'], reverse=True))
    ])

    system_message = '''Ты - медицинская экспертная система. Твоя задача - найти наиболее релевантные фрагменты контекста и дать научно обоснованный ответ.

        КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
        1. Сравни вопрос пользователя с вопросами во фрагментах 
        2. Выбери фрагменты, где вопросы максимально близки по смыслу и намерению
        3. Используй для ответа ТОЛЬКО выбранные релевантные фрагменты
        4. Включай в reference ТОЛЬКО URL из использованных фрагментов

        ПРАВИЛА СОСТАВЛЕНИЯ ОТВЕТА:
        1. Ответ должен быть структурированным и логичным
        2. Включай все важные детали: цифры, факты, научные термины
        3. Используй научный стиль изложения
        4. Сохраняй точность формулировок из источников
        5. Если есть противоречия между фрагментами - укажи на них
        6. При наличии статистики или исследований - приводи их результаты
        7. Избегай обобщений без подтверждения данными

        ФОРМАТ ОТВЕТА:
        {
        "answer": "Подробный научный ответ на основе релевантных фрагментов",
        "reference": "URL использованных фрагментов"
        }'''

    user_message = f'''КОНТЕКСТ:
        {context_text}

        ВОПРОС ПОЛЬЗОВАТЕЛЯ:
        {query}

        ПРОЦЕСС:
        1. Найди фрагменты с максимально похожими вопросами
        2. Проанализируй информацию в этих фрагментах
        3. Составь структурированный научный ответ
        4. Включи все важные данные и детали
        5. Добавь в reference ТОЛЬКО URL использованных фрагментов
        6. Проверь точность и полноту ответа

        Верни JSON с полями "answer" и "reference"'''
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Ошибка при генерации ответа: {str(e)}")
        return None

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
    
    # Получаем эмбеддинг один раз
    query_embedding = get_embedding(query)
    if not query_embedding:
        await update.message.reply_text("Извините, произошла ошибка при обработке вопроса.")
        return
    
    # Ищем среди всех ответов
    relevant_context = get_relevant_context(query, query_embedding=query_embedding, include_generated=True)
    
    # Если контекст пустой, значит вопрос не по теме
    if not relevant_context:
        await update.message.reply_text("Извините, в базе знаний нет релевантной информации по вашему вопросу.")
        return
    
    # Если есть ответ с высокой релевантностью - возвращаем его
    if relevant_context[0]['relevance'] >= DIRECT_ANSWER_RELEVANCE:
        most_relevant = relevant_context[0]
        emoji = "🚀" if most_relevant['is_generated'] else "📖"
        references = "\n".join(most_relevant['reference'].split())
        response = f"{emoji} {most_relevant['answer']}\n\nИсточники:\n{references}"
        
    # Если нет ответа с высокой релевантностью - генерируем новый на основе только оригинальных ответов
    else:
        original_context = get_relevant_context(query, query_embedding=query_embedding, include_generated=False)
        if not original_context:
            await update.message.reply_text("Извините, в базе знаний нет достаточно релевантной информации по вашему вопросу.")
            return
            
        generated = generate_response(query, original_context)
        if not generated:
            await update.message.reply_text("Извините, произошла ошибка при генерации ответа.")
            return
            
        response_data = json.loads(generated)
        # Используем уже полученный эмбеддинг для сохранения
        save_generated_answer(query, response_data["answer"], response_data["reference"])
        references = "\n".join(response_data["reference"].split())
        response = f"🧠 {response_data['answer']}\n\nИсточники:\n{references}"
    
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