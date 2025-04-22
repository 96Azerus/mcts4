# Dockerfile v1.2
# Используем официальный образ Python 3.11
FROM python:3.11

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей сначала, чтобы использовать кэш Docker
COPY requirements.txt requirements.txt

# Устанавливаем системные зависимости, если они нужны (например, для C-расширений)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Обновляем pip и устанавливаем зависимости Python без кэша
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Создаем непривилегированного пользователя и группу для запуска приложения
# Используем /sbin/nologin для большей безопасности
RUN adduser --system --no-create-home --group --disabled-password --gecos "" appuser

# Копируем все файлы приложения (включая src, tests, templates и т.д.)
COPY . .

# Устанавливаем владельца для всего /app, чтобы appuser мог писать логи или кэш (если нужно)
# Делаем это после копирования всех файлов
RUN chown -R appuser:appuser /app

# Устанавливаем PYTHONPATH, чтобы Python находил модули в /app (где лежит src)
ENV PYTHONPATH=/app

# Переключаемся на непривилегированного пользователя
USER appuser

# Указываем Flask, где искать приложение (app.py в корне /app)
ENV FLASK_APP=app.py
# Переменные PORT и WEB_CONCURRENCY будут установлены средой выполнения (например, Render, Docker Compose)

# Команда по умолчанию для запуска приложения с использованием Gunicorn
# Gunicorn будет запускаться от имени appuser
# Используем exec для того, чтобы Gunicorn стал PID 1 в контейнере
# Используем переменные окружения PORT и WEB_CONCURRENCY с дефолтными значениями
# Добавляем --timeout для долгих MCTS ходов (120 секунд)
# Используем --log-level info для стандартного уровня логирования
CMD exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers ${WEB_CONCURRENCY:-2} --timeout 120 --log-level info app:app
