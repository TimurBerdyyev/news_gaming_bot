
import feedparser
import telegram
import schedule
import time
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient
from langdetect import detect
import os
import json
import logging
import signal
import threading
import re
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import requests
from telegram.error import RetryAfter, TimedOut, NetworkError
from dotenv import load_dotenv
import asyncio
import html as html_lib

# Загрузка переменных окружения из .env
load_dotenv()

# Константы из окружения
BOT_TOKEN = os.getenv('BOT_TOKEN', '')
CHANNEL_ID = os.getenv('CHANNEL_ID', '')
HF_TOKEN = os.getenv('HF_TOKEN', '')
GENERATE_VIA_HF = os.getenv('GENERATE_VIA_HF', '0')
USE_GROQ = os.getenv('USE_GROQ', '0')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')

# Валидация обязательных переменных
if not BOT_TOKEN:
    raise RuntimeError('Не указан BOT_TOKEN в переменных окружения')
if not CHANNEL_ID:
    raise RuntimeError('Не указан CHANNEL_ID в переменных окружения')

# Настройки
MAX_TELEGRAM_MESSAGE = 4096
MAX_CAPTION = 1024
SUMMARY_MAX_LEN = 2000
REQUEST_TIMEOUT_SEC = 20
MAX_RETRIES = 3
BACKOFF_BASE_SEC = 2
LAST_POSTS_FILE = 'last_posts.json'
USER_AGENT = 'NewsGameBot/1.0 (+https://t.me/your_bot)'

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Клиент Hugging Face (по умолчанию отключён, если флаг не выставлен)
client = InferenceClient(token=HF_TOKEN) if (HF_TOKEN and GENERATE_VIA_HF == '1') else None

# Список RSS-фидов
rss_feeds = [
    'https://www.gameinformer.com/rss',
    'https://www.playground.ru/rss/news.xml',
]

# Хранилище для последних новостей
last_posts = {}

stop_event = threading.Event()

def load_last_posts():
    try:
        if os.path.exists(LAST_POSTS_FILE):
            with open(LAST_POSTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        logging.warning(f'Не удалось загрузить {LAST_POSTS_FILE}: {e}')
    return {}

def save_last_posts():
    try:
        with open(LAST_POSTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(last_posts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f'Не удалось сохранить {LAST_POSTS_FILE}: {e}')

def normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        query_params = [(k, v) for k, v in parse_qsl(parsed.query) if not k.lower().startswith('utm_')]
        clean_query = urlencode(query_params)
        normalized = parsed._replace(query=clean_query, fragment='')
        return urlunparse(normalized)
    except Exception:
        return url

def short_source_anchor(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return f'<a href="{html_lib.escape(url)}">{html_lib.escape(netloc)}</a>'
    except Exception:
        return html_lib.escape(url)

def split_message(text: str, limit: int = MAX_TELEGRAM_MESSAGE) -> list:
    if len(text) <= limit:
        return [text[:limit]]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        chunk = text[start:end]
        last_nl = chunk.rfind('\n')
        last_sp = chunk.rfind(' ')
        cut = max(last_nl, last_sp)
        if cut > 0:
            parts.append(chunk[:cut].rstrip())
            start += cut
        else:
            parts.append(chunk)
            start = end
    return parts

def request_with_retries(url: str) -> requests.Response | None:
    headers = {'User-Agent': USER_AGENT}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
            if resp.status_code == 200:
                return resp
            logging.warning(f'HTTP {resp.status_code} для {url} (попытка {attempt}/{MAX_RETRIES})')
        except requests.RequestException as e:
            logging.warning(f'Ошибка запроса {url}: {e} (попытка {attempt}/{MAX_RETRIES})')
        time.sleep(BACKOFF_BASE_SEC ** attempt)
    return None

def is_game_related(title: str, summary: str) -> bool:
    game_keywords = [
        'game', 'gaming', 'video game', 'nintendo', 'playstation', 'xbox', 
        'pc game', 'steam', 'epic games', 'assassin', 'battlefield', 
        'final fantasy', 'hollow knight', 'dying light'
    ]
    non_game_keywords = [
        'blu-ray', 'movie', 'film', 'collection', 'lego', 'soundcore', 
        'anker', 'casino', 'scorsese', 'batman anniversary', 'zombie movie'
    ]
    title_lower = title.lower()
    summary_lower = summary.lower()
    has_game_keyword = any(keyword in title_lower or keyword in summary_lower for keyword in game_keywords)
    has_non_game_keyword = any(keyword in title_lower or keyword in summary_lower for keyword in non_game_keywords)
    return has_game_keyword and not has_non_game_keyword

async def send_telegram_message(bot: telegram.Bot, chat_id: str, text: str):
    logging.info(f"Отправка сообщения в {chat_id}: {text[:50]}...")
    try:
        await bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=False)
        logging.info("Сообщение успешно отправлено")
        await asyncio.sleep(2)  # Задержка 2 секунды
    except Exception as e:
        logging.error(f"Ошибка отправки: {e}")
        raise e

def get_image_url(entry) -> str | None:
    try:
        # media_content
        media = getattr(entry, 'media_content', None)
        if media and isinstance(media, list):
            for m in media:
                u = m.get('url') if isinstance(m, dict) else None
                if u:
                    return u
        # media_thumbnail
        thumbs = getattr(entry, 'media_thumbnail', None)
        if thumbs and isinstance(thumbs, list):
            for m in thumbs:
                u = m.get('url') if isinstance(m, dict) else None
                if u:
                    return u
        # enclosures
        enclosures = getattr(entry, 'enclosures', None)
        if enclosures and isinstance(enclosures, list):
            for enc in enclosures:
                u = enc.get('href') if isinstance(enc, dict) else None
                if u and any(u.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    return u
        # content images as fallback
        content = entry.get('content', []) if isinstance(entry, dict) else []
        for c in content:
            val = c.get('value', '')
            if 'img ' in val:
                # простая эвристика
                import re as _re
                m = _re.search(r'src=["\']([^"\']+)', val)
                if m:
                    return m.group(1)
    except Exception:
        return None
    return None

def process_news(title, summary, link):
    try:
        safe_link = normalize_url(link or '')
        if not client:
            return f"<b>📰 {html_lib.escape(title)}</b>\n\nИсточник: {short_source_anchor(safe_link)}"
        
        summary = (summary or '')[:SUMMARY_MAX_LEN]
        text_for_lang = f"{title}\n{summary}".strip()
        need_translate = False
        if text_for_lang and len(text_for_lang) >= 20:
            try:
                lang = detect(text_for_lang)
                need_translate = (lang != 'ru')
            except Exception:
                need_translate = False

        if need_translate:
            try:
                translation = client.translation(
                    text=summary or title,
                    model="Helsinki-NLP/opus-mt-en-ru",
                )
                summary = translation['translation_text'] if isinstance(translation, dict) else translation
            except Exception as e:
                logging.warning(f'Ошибка перевода HF: {e}')
                summary = summary or title

        summary_text = summary
        try:
            summarization = client.summarization(
                text=summary,
                model="facebook/bart-large-cnn",
            )
            summary_text = summarization['summary_text'] if isinstance(summarization, dict) else summarization
        except Exception as e:
            logging.warning(f'Ошибка саммари HF: {e}')
            summary_text = summary
        
        message = f"<b>📰 {html_lib.escape(title)}</b>\n\n{html_lib.escape(summary_text)}\n\nИсточник: {short_source_anchor(safe_link)}"
        return message
    except Exception as e:
        logging.error(f"Ошибка HF API: {e}")
        return f"<b>📰 {html_lib.escape(title)}</b>\n\nИсточник: {short_source_anchor(link or '')}"

def generate_post_via_hf(title: str, summary: str) -> str | None:
    if not client:
        return None
    try:
        prompt = (
            "Сформулируй короткий новостной пост (2-4 предложения) по теме видеоигр на русском. "
            "Избегай клише и воды, будь информативным и нейтральным. "
            f"Заголовок: {title}\nТекст: {summary[:800]}"
        )
        out = client.text_generation(
            prompt=prompt,
            model="tiiuae/falcon-7b-instruct",
            max_new_tokens=120,
            temperature=0.6,
            top_p=0.9,
        )
        if isinstance(out, dict):
            return out.get('generated_text')
        return str(out)
    except Exception as e:
        logging.warning(f"HF text_generation ошибка: {e}")
        return None

def generate_post_via_groq(title: str, summary: str) -> str | None:
    if USE_GROQ != '1' or not GROQ_API_KEY:
        return None
    try:
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json',
        }
        prompt = (
            "Сформулируй короткий новостной пост (2-4 предложения) по теме видеоигр на русском. "
            "Избегай клише и воды, будь информативным и нейтральным. "
            f"Заголовок: {title}\nТекст: {summary[:800]}"
        )
        payload = {
            'model': GROQ_MODEL,
            'messages': [
                {'role': 'system', 'content': 'Ты помощник-редактор новостей об играх. Пиши кратко и по делу.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.6,
            'top_p': 0.9,
            'max_tokens': 200,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        if resp.status_code != 200:
            logging.warning(f'Groq API {resp.status_code}: {resp.text[:200]}')
            return None
        data = resp.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content')
        return content
    except Exception as e:
        logging.warning(f'Groq API ошибка: {e}')
        return None

async def fetch_and_post_news():
    bot = telegram.Bot(token=BOT_TOKEN)
    async with bot:
        news_sent = 0
        for feed_url in rss_feeds:
            if news_sent >= 1:
                break
            resp = request_with_retries(feed_url)
            if not resp:
                continue
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                if news_sent >= 1:
                    break
                try:
                    if 'published_parsed' in entry and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                        if pub_date < datetime.now() - timedelta(hours=24):
                            continue

                    title = entry.get('title', 'Без заголовка')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')

                    post_key = f"{title}_{normalize_url(link)}"
                    if post_key in last_posts:
                        continue

                    message_html = process_news(title, summary, link)
                    image_url = get_image_url(entry)

                    # Сначала пробуем Groq (если включён), затем HF (если включён)
                    generated = generate_post_via_groq(title, summary)
                    if not generated and GENERATE_VIA_HF == '1':
                        generated = generate_post_via_hf(title, summary) if 'generate_post_via_hf' in globals() else None

                    if generated:
                        safe_link = normalize_url(link or '')
                        message_html = (
                            f"<b>📰 {html_lib.escape(title)}</b>\n\n"
                            f"{html_lib.escape(generated)}\n\n"
                            f"Источник: {short_source_anchor(safe_link)}"
                        )

                    if image_url:
                        caption = message_html
                        if len(caption) > MAX_CAPTION:
                            caption = caption[:MAX_CAPTION - 1] + '…'
                        await bot.send_photo(chat_id=CHANNEL_ID, photo=image_url, caption=caption, parse_mode='HTML')
                    else:
                        for part in split_message(message_html, MAX_TELEGRAM_MESSAGE):
                            await bot.send_message(chat_id=CHANNEL_ID, text=part, parse_mode='HTML', disable_web_page_preview=False)

                    last_posts[post_key] = True
                    save_last_posts()
                    logging.info(f"Опубликовано: {title}")
                    news_sent += 1
                except Exception as e:
                    logging.error(f'Ошибка обработки записи фида: {e}')
def handle_signal(signum, frame):
    logging.info(f'Получен сигнал {signum}. Завершаем работу...')
    stop_event.set()

async def main():
    global last_posts
    last_posts = load_last_posts()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logging.info("Бот запущен. Проверяет новости каждые 30 минут.")

    # Раз в 6 часов
    schedule.clear()
    schedule.every(6).hours.do(lambda: asyncio.create_task(fetch_and_post_news()))

    if os.getenv('AUTO_START') == '1':
        try:
            await fetch_and_post_news()
        except Exception as e:
            logging.error(f'Ошибка при первичном запуске: {e}')

        while not stop_event.is_set():
            schedule.run_pending()
            await asyncio.sleep(1)

        save_last_posts()
        logging.info('Бот остановлен.')
    else:
        logging.info('AUTO_START != 1 — автозапуск отключён. Скрипт завершает работу.')

if __name__ == '__main__':
    asyncio.run(main())
    