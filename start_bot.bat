@echo off
title Telegram Bot + Llama3 Launcher
color 0A

:: 1. Проверка установки Ollama
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama is not downloaded! Download from https://ollama.ai/
    pause
    exit /b
)

:: 2. Запуск Ollama в фоновом режиме
echo Activating Ollama...
start /B "" ollama serve

:: 3. Активация окружения Conda и запуск бота
echo Activating Telegram bot...
call conda activate Text_normalization
python ".\test_tg_bot.py"

:: 4. Остановка Ollama при завершении (если нужно)
taskkill /IM ollama.exe /F >nul 2>&1
echo Script work ended
pause