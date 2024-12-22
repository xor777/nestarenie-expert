# PoC of RAG Chatbot for the "Nestarenie" project

## Перед запуском

0. Установите зависимости: `pip install -r requirements.txt`
1. Подготовьте dataset.csv (формат: Вопрос, Ответ, Ссылка)
2. Создайте `.env` файл с необходимыми переменными окружения:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   TEMPERATURE=0.3
   MIN_RELEVANCE=0.5
   MAX_TOKENS=8000
   ```

## Использование

1. Загрузите базу знаний в ChromaDB: `python load_dataset.py`
2. Запустите бота: `python telegram_chat.py`
