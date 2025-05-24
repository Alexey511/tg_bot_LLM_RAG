 A Telegram bot that uses Retrieval-Augmented Generation (RAG) to search a knowledge base of user-submitted texts, leveraging embeddings and a local Llama model for semantic search and response generation.
 
 ## Features
 - Save text to a knowledge base with `/save_on`.
 - Search for relevant documents with `/search <query>`.
 - Uses `paraphrase-multilingual-mpnet-base-v2` for embeddings and ChromaDB for vector search.
 - Query rephrasing with Llama for improved paraphrase handling.
 
 ## Technologies
 - Python 3.9+
 - Sentence Transformers (`mpnet-base-v2`)
 - ChromaDB
 - Llama (via Ollama)
 - Telegram Bot API
 
 ## Setup
 1. Clone the repository: `git clone <repo-url>`
 2. Install dependencies: `pip install -r requirements.txt`
 3. Set up `.env` with `TELEGRAM_TOKEN` and `OPENAI_API_KEY` (dummy for Ollama).
 4. Run Llama server: `ollama run llama3`
 5. Start the bot: `python telegram_bot.py`
 
 ## Usage
 - `/save_on`: Enable save mode.
 - Send text to save (e.g., “В нашей компании работает 35 сотрудников”).
 - `/save_off`: Disable save mode.
 - `/search Сколько сотрудников в вашей компании?`: Search the knowledge base.
 
 ## Challenges and Solutions
 - **Issue**: Large negative similarity scores.
 - **Solution**: Normalized embeddings and reset ChromaDB.
 - **Issue**: Low similarity for paraphrased queries.
 - **Solution**: Added query rephrasing with Llama and switched to `mpnet-base-v2`.
 
 ## Future Improvements
 - Fine-tune embedding model for Russian paraphrases.
 - Add evaluation metrics (e.g., precision@1).
 - Optimize for scalability with batch embedding.
 
 ## Demo
 [Insert screenshot or link to video]