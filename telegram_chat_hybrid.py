import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict, TypedDict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import json
from dataclasses import dataclass

load_dotenv()

@dataclass
class Config:
    openai_api_key: str
    telegram_token: str
    temperature: float
    min_relevance: float
    max_input_tokens: int
    direct_answer_relevance: float
    embedding_model: str
    generation_model: str
    
    @classmethod
    def from_env(cls) -> 'Config':
        load_dotenv()
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("Не найден ключ OPENAI_API_KEY в переменных окружения")
            
        telegram_token = os.getenv('TELEGRAM_TOKEN')
        if not telegram_token:
            raise ValueError("Не найден токен TELEGRAM_TOKEN в переменных окружения")
            
        return cls(
            openai_api_key=openai_api_key,
            telegram_token=telegram_token,
            temperature=float(os.getenv('TEMPERATURE', '0.1')),
            min_relevance=float(os.getenv('MIN_RELEVANCE', '0.9')),
            max_input_tokens=int(os.getenv('MAX_INPUT_TOKENS', '1000')),
            direct_answer_relevance=float(os.getenv('DIRECT_ANSWER_RELEVANCE', '0.98')),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            generation_model=os.getenv('GENERATION_MODEL', 'gpt-4')
        )

config = Config.from_env()

client_openai = OpenAI(api_key=config.openai_api_key)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("questions")

class ContextItem(TypedDict):
    question: str
    answer: str
    reference: str
    relevance: float
    is_generated: bool

class GeneratedResponse(TypedDict):
    answer: str
    reference: str

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        text = " ".join(text.split())
        
        if len(text) > config.max_input_tokens * 4:
            print("Предупреждение: текст слишком длинный, будет использована только его часть")
            text = text[:config.max_input_tokens * 4]
            
        response = client_openai.embeddings.create(
            model=config.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Ошибка при получении эмбеддинга: {str(e)}")
        return None

def save_generated_answer(question: str, answer: str, reference: str, embedding: Optional[List[float]] = None) -> None:
    try:
        if not embedding:
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

def get_relevant_context(query: str, query_embedding: List[float], include_generated: bool = True) -> List[ContextItem]:
    print(f"searching... [pre-generated {'included' if include_generated else 'excluded'}]")
    
    try:
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": 5,
            "include": ["documents", "metadatas", "distances"]
        }
        if not include_generated:
            query_params["where"] = {"is_generated": False}
            
        results = collection.query(**query_params)
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
            print(f"relevance: {relevance} | skipped (relevance): {question}")
            continue
            
        print(f"relevance: {relevance} | added: {question}")
        context.append({
            "question": question,
            "answer": metadata["answer"],
            "reference": metadata["reference"],
            "relevance": relevance,
            "is_generated": metadata.get('is_generated', False)
        })
    return context

def generate_response(query: str, context: List[ContextItem]) -> Optional[str]:
    if not context:
        return None
    
    context_text = "\n\n".join([
        f"ФРАГМЕНТ #{i+1}\nВОПРОС:\n{c['question']}\nОТВЕТ:\n{c['answer']}\nURL:\n{c['reference']}"
        for i, c in enumerate(sorted(context, key=lambda x: x['relevance'], reverse=True))
    ])

    system_message = '''Ты - медицинская экспертная система. Твоя задача - предоставлять научно точные, хорошо структурированные ответы на основе релевантных фрагментов контекста.

        КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
        1. Сравни вопрос пользователя с вопросами во фрагментах
        2. Выбери фрагменты, где вопросы максимально близки по смыслу
        3. Используй ТОЛЬКО информацию из выбранных фрагментов
        4. В reference включай ТОЛЬКО URL тех фрагментов, чья информация вошла в ответ
        5. НИКОГДА не додумывай информацию - если данных недостаточно, честно скажи об этом

        ПРАВИЛА СОСТАВЛЕНИЯ ОТВЕТА:
        1. Полнота и структура:
        - Начни с прямого ответа на вопрос
        - Раскрой все важные аспекты темы
        - Включи существенные детали, цифры, термины
        - Добавь важные уточнения и нюансы
        - Сохраняй логическую связность

        2. Научная точность:
        - Используй точные формулировки из источников
        - Приводи конкретные результаты исследований
        - Указывай статистические данные
        - При наличии противоречий - отрази разные позиции

        3. Информативность:
        - Каждое предложение должно нести значимую информацию
        - Избегай общих фраз и самоочевидных утверждений
        - Не повторяйся, если это не нужно для понимания
        - Раскрывай суть научных терминов при необходимости

        ФОРМАТ ОТВЕТА:
        {
        "answer": "Информативный, хорошо структурированный ответ с необходимой детализацией",
        "reference": "URL только тех фрагментов, информация из которых использована в ответе"
        }'''

    user_message = f'''КОНТЕКСТ:
        {context_text}

        ВОПРОС:
        {query}

        ПРОЦЕСС:
        1. Найди фрагменты с максимально релевантными вопросами
        2. Проверь достаточность информации
        3. Составь полный, структурированный ответ:
        - Основные факты и данные
        - Важные детали и нюансы
        - Научные термины и их объяснения
        - Результаты исследований
        4. Убедись, что каждый включенный URL соответствует использованной информации
        5. Проверь информативность каждого предложения

        Верни JSON с полями "answer" и "reference"'''
    
    try:
        response = client_openai.chat.completions.create(
            model=config.generation_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=config.temperature,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Ошибка при генерации ответа: {str(e)}")
        return None

def format_references(reference: str) -> str:
    if not reference or reference.isspace():
        return ""
    return f"\n\nИсточники:\n{'\n'.join(reference.split())}"

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
    if relevant_context[0]['relevance'] >= config.direct_answer_relevance:
        most_relevant = relevant_context[0]
        emoji = "🚀" if most_relevant['is_generated'] else "📖"
        response = f"{emoji} {most_relevant['answer']}{format_references(most_relevant['reference'])}"
        
    # Если нет ответа с высокой релевантностью - генерируем новый
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
        save_generated_answer(
            question=query, 
            answer=response_data["answer"], 
            reference=response_data["reference"],
            embedding=query_embedding
        )
        response = f"🧠 {response_data['answer']}{format_references(response_data['reference'])}"
    
    await update.message.reply_text(response)

def main() -> None:
    if not os.path.exists("./chroma_db"):
        print("База данных не найдена. Сначала запустите load_dataset.py")
        return

    application = Application.builder().token(config.telegram_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main() 