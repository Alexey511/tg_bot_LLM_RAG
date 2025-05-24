@echo off
title Telegram Bot + Llama3 Launcher
color 0A

:: 1. Проверка установки Ollama
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Ollama не установлен! Скачайте с https://ollama.ai/
    pause
    exit /b
)

:: 2. Запуск Ollama в фоновом режиме
echo 🚀 Запускаю Ollama сервер...
start /B "" ollama serve

:: 3. Активация окружения Conda и запуск бота
echo 🤖 Запускаю Telegram-бота...
call conda activate Text_normalization
python "C:\Users\User\Documents\Progs\Projects\tg_bot_llm_rag\test_tg_bot.py"

:: 4. Остановка Ollama при завершении (если нужно)
taskkill /IM ollama.exe /F >nul 2>&1
echo ✅ Скрипт завершен
pause