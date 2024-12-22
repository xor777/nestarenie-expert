# PoC of RAG Chatbot for the "Nestarenie" project

## Перед запуском

0. Установите зависимости: `pip install -r requirements.txt`
1. Подготовьте dataset.csv (формат: Вопрос, Ответ, Ссылка)
2. Создайте `.env` файл с необходимыми переменными окружения:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   TEMPERATURE=0.3 # температура для генеративной модели (0 - консервативно, 1 - агрессивно)
   MIN_RELEVANCE=0.5 # минимальное значение релевантности для ответа для поиска в векторной базе (0 - смотрим все, 1 - смотрим только 100% релевантные)
   MAX_TOKENS=8000 # максимальное количество токенов в запросе от пользователя
   ```

## Использование

1. Загрузите базу знаний в ChromaDB: `python load_dataset.py`
2. Запустите бота: `python telegram_chat.py`
