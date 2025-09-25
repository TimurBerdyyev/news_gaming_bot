# Game News Telegram Bot

Бот публикует новости об играх из RSS-ленты каждые 6 часов, добавляет кликаемую ссылку-источник, изображение (если доступно), умеет генерировать короткий текст поста через Groq (или Hugging Face) и добавлять информацию о скидках на игру.

## Требования
- Python 3.11+ (рекомендовано 3.12)
- Установленные зависимости из `requirements.txt`

## Установка
```bash
pip install -r requirements.txt
```

## Настройка `.env`
Создайте файл `.env` рядом с `main.py`/`main2.py` (можно взять за основу `.env.example`) и заполните:
```
BOT_TOKEN=...               # токен бота от @BotFather
CHANNEL_ID=@your_channel    # @username канала или -100XXXXXXXXXX
# Для Groq (опционально)
USE_GROQ=1
GROQ_API_KEY=...            # ключ с https://console.groq.com/keys
GROQ_MODEL=llama-3.1-8b-instant
# Для Hugging Face (опционально)
GENERATE_VIA_HF=0
HF_TOKEN=...                # токен с правами Inference
```

## Запуск
- Разовый запуск с автопостингом:
```bash
AUTO_START=1 python3 main.py
```
- В фоне с логами:
```bash
mkdir -p logs
AUTO_START=1 python3 main.py >> logs/bot.out 2>&1 &
tail -n 200 -f logs/bot.out
```

## Как это работает
- Каждые 6 часов бот берёт по 1 новости из RSS (`rss_feeds` в коде),
- извлекает изображение (media/enclosure/thumbnail/HTML),
- формирует текст:
  - при `USE_GROQ=1` — генерирует краткий пост через Groq API,
  - иначе (при `GENERATE_VIA_HF=1`) — использует Hugging Face (перевод+саммари),
  - иначе — отправляет базовый текст с источником.
- Если новость про игры — пытается найти лучшую скидку через CheapShark и добавляет в пост.

## Полезные команды
- Остановить все запуски:
```bash
pkill -f "python.*main.py"
```
- Проверить переменные окружения, если бот жалуется:
```bash
env | grep -E 'BOT_TOKEN|CHANNEL_ID|GROQ|HF_TOKEN'
```

## Частые проблемы
- ModuleNotFoundError/urllib3: убедитесь, что установлен `python-telegram-bot>=21,<22` и `httpx`.
- 403 от Hugging Face: нужен токен с правами Inference. Иначе отключите `GENERATE_VIA_HF`.
- Нет публикаций: проверьте `CHANNEL_ID` и права бота в канале (бот должен быть администратором).

## Кастомизация
- Измените список фидов в `rss_feeds`.
- Настройте интервал в `schedule.every(6).hours`.
- Порог скидок — `MIN_DISCOUNT_PERCENT` в `main.py`.

