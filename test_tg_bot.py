import os
import sys
import asyncio
import psutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import Conflict

from openai import OpenAI
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# Выберите модель эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768
# Альтернативы:
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_DIM = 384
# EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"
# EMBEDDING_DIM = 512

# Загрузка конфигурации
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Модель для эмбеддингов
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# Функция для нормализации эмбеддингов
def normalize_embedding(embedding):
    """Normalize embedding to unit length (L2 norm = 1)."""
    embedding = np.array(embedding)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return (embedding / norm).tolist()
    return embedding.tolist()

# --- Инициализация ChromaDB ---
database = Client(settings=Settings(
    persist_directory="./db",
    allow_reset=True,
    is_persistent=True  # Автосохранение включено
))

collection = database.get_or_create_collection(
    name="docs",
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}  # Явно указываем тип эмбеддинга
)

# --- Настройка локального подключения к Ollama ---
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Фиктивный ключ (не требуется)
)

async def llama_response(prompt: str) -> str:
    """Запрос к локальной Llama 3"""
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Ошибка: {str(e)}"

async def rephrase_query(query: str) -> str:
    """Переформулировать запрос в краткий, прямой и ясный вопрос"""
    prompt = (
        "Переформулируй запрос в краткий и ясный вопрос на русском языке, полностью сохраняя его исходный смысл. "
        "Используй нейтральный, профессиональный язык, подходящий для делового контекста, и избегай неформальных слов (например, 'люди' вместо 'сотрудники') или лишних фраз. "
        "Убедись, что тематика запроса (например, цель компании, количество сотрудников) остается неизменной. Всегда отвечай только на русском языке.\n"
        "Примеры:\n"
        "1. 'Какое количество лиц в вашем коллективе?' → 'Сколько сотрудников в вашей компании?'\n"
        "2. 'Какова цель вашей организации?' → 'Какова цель вашей компании?'\n"
        "3. 'Кто лучший работник месяца?' → 'Кто лучший сотрудник месяца в вашей компании?'\n"
        "4. 'Как давно существует ваша фирма?' → 'Каков возраст вашей компании?'\n"
        "5. 'Какое имя носит ваша контора?' → 'Как называется ваша компания?'\n"
        f"Запрос: '{query}'"
    )
    response = await llama_response(prompt)
    #print(f"Rephrased query: '{query}' -> '{response}'")
    return response


# --- Обработчики Telegram ---
async def set_commands(app):
    commands = [
        BotCommand("start", "Начало работы"),
        BotCommand("search", "Поиск по базе знаний"),
        BotCommand("save_on", "Режим сохранения (вкл)"),
        BotCommand("save_off", "Режим сохранения (выкл)"),
        BotCommand("delete", "Удалить документ по ID")
    ]
    await app.bot.set_my_commands(commands)

async def post_init(application):
    """Функция инициализации после запуска"""
    await set_commands(application)

async def start(update: Update, context):
    await update.message.reply_text(
        "🤖 Я бот с LLama AI. Напиши мне что-нибудь!"
    )

async def save_mode_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Активирует режим сохранения следующего сообщения"""
    context.user_data["save_mode"] = True
    await update.message.reply_text("✅ Режим сохранения включён. Отправь текст, который нужно запомнить.")

async def save_mode_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Деактивирует режим сохранения следующего сообщения"""
    context.user_data["save_mode"] = False
    await update.message.reply_text("❌ Режим сохранения выключён. Теперь я обычный чат-бот.")

async def search_docs(update: Update, context):
    """Поиск документов с ближайшим контекстом к запросу"""
    if not context.args:
        await update.message.reply_text("Введите запрос: /search <ваш запрос>")
        return
    
    query = " ".join(context.args)
    query = await rephrase_query(query)
    
    # Нормализация эмбеддинга запроса
    query_embedding = normalize_embedding(embedding_func([query])[0])
    norm = np.linalg.norm(query_embedding)
    #print(f"Norm of query embedding for '{query[:50]}...': {norm}")
    
    # Ищем 5 самых релевантных документов
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    # Логирование расстояний для отладки
    # for dist in results["distances"][0]:
    #     print(f"Distance: {dist}")
    #     if dist < 0 or dist > 2:
    #         print(f"Warning: distance {dist} is out of expected range for cosine distance")
    
    if not results["documents"]:
        await update.message.reply_text("Ничего не найдено 😔")
        return
    
    response = ["🔍 Результаты поиска:"]
    for i, (doc, meta, dist, doc_id) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0]
    ), 1):
        user = meta.get("user", "unknown")
        similarity = max(0, min(1, 1 - dist))  # Зажимаем схожесть в [0, 1]
        response.append(
            f"{i}. ID: `{doc_id}`\n"
            f"   👤 {user}\n"
            f"   📝 {doc[:200]}...\n"
            f"   📊 Схожесть: {similarity:.2f}\n"
        )
    
    await update.message.reply_text(
        "\n".join(response)[:4000],
        parse_mode="Markdown"
    )

async def delete_doc(update: Update, context):
    """Удаление документа по его ID"""
    doc_id = context.args[0] if context.args else None
    if not doc_id:
        await update.message.reply_text("Укажите ID: /delete <id> (например, /delete doc_123)")
        return
    
    collection.delete(ids=[doc_id])
    await update.message.reply_text(f"Документ {doc_id} удалён")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    user = update.effective_user
    text = update.message.text

    if context.user_data.get("save_mode"):
        doc_id = f"doc_{user.id}_{datetime.now().timestamp()}"
        embedding = normalize_embedding(embedding_func([text])[0])
        norm = np.linalg.norm(embedding)
        #print(f"Norm of saved embedding for text '{text[:50]}...': {norm}")
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "user": user.username if user.username else str(user.id),
                "source": "telegram",
                "timestamp": datetime.now().isoformat()
            }]
        )
        await update.message.reply_text("💾 Сохранено! Режим сохранения активен. Используйте /save_off для выхода.")
        return

    # Initialize conversation history
    if "conversation_history" not in context.user_data:
        context.user_data["conversation_history"] = []

    # Add current query to history
    context.user_data["conversation_history"].append({"role": "user", "content": text})
    # Limit history to 5 messages
    context.user_data["conversation_history"] = context.user_data["conversation_history"][-5:]

    query = await rephrase_query(text)
    
    query_embedding = normalize_embedding(embedding_func([query])[0])
    #norm = np.linalg.norm(query_embedding)
    #print(f"Norm of query embedding for '{query[:50]}...': {norm}")
    
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=15,
        include=["documents", "distances"]
    )

    if not search_results["distances"][0]:
        response = await llama_response(f"Вопрос: {text}")
        context.user_data["conversation_history"].append({"role": "assistant", "content": response})
        await update.message.reply_text(response[:4000])
        return

    # Dynamic search logic
    scored_docs = sorted(
        [(1 - dist, doc) for doc, dist in zip(search_results["documents"][0], search_results["distances"][0])],
        key=lambda x: x[0], 
        reverse=True
    )

    max_similarity = scored_docs[0][0]
    dynamic_threshold = max(0.1, max_similarity - 0.1)

    context_parts = []
    total_length = 0
    MAX_CONTEXT_LENGTH = 3000
    total_docs = len(scored_docs)

    for similarity, doc in scored_docs:
        if similarity < dynamic_threshold:
            continue
        doc_length = len(doc)
        if total_length + doc_length > MAX_CONTEXT_LENGTH:
            remaining_space = MAX_CONTEXT_LENGTH - total_length
            if remaining_space > 50:
                context_parts.append(doc[:remaining_space])
            break
        context_parts.append(doc)
        total_length += doc_length

    context_text = "\n\n".join(context_parts)
    
    # Build prompt with conversation history
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context.user_data["conversation_history"][:-1]])
    prompt = (
        f"Контекст (отобрано {len(context_parts)}/{total_docs} документов, "
        f"порог схожести: {dynamic_threshold:.2f}):\n{context_text}\n\n"
        f"История разговора:\n{history_text}\n\n"
        f"Вопрос: {text}\n\n"
        "Ответь строго по контексту, учитывая историю разговора, избегая домыслов."
    )
    
    response = await llama_response(prompt)
    context.user_data["conversation_history"].append({"role": "assistant", "content": response})
    await update.message.reply_text(response[:4000])

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("save_on", save_mode_on))
    app.add_handler(CommandHandler("save_off", save_mode_off))
    app.add_handler(CommandHandler("search", search_docs))
    app.add_handler(CommandHandler("delete", delete_doc))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    print("=== Начало работы ===")
    main()
    print("=== Завершение работы ===")