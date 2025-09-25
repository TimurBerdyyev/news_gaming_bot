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
from operator import itemgetter  # Для сортировки скидок

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
MIN_DISCOUNT_PERCENT = 10  # Минимальная скидка для показа

# Логирование
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
POST_INTERVAL_HOURS = int(os.getenv('POST_INTERVAL_HOURS', '6'))
MIN_DISCOUNT_PERCENT = int(os.getenv('MIN_DISCOUNT_PERCENT', str(MIN_DISCOUNT_PERCENT)))
MAX_DEALS = int(os.getenv('MAX_DEALS', '3'))
DEAL_TTL_HOURS = int(os.getenv('DEAL_TTL_HOURS', '12'))

# Хранилища ETag/Last-Modified для RSS
rss_etags: dict[str, str] = {}
rss_last_modified: dict[str, str] = {}

# Логирование
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# TTL-кэш скидок
from time import monotonic
_deal_timestamps: dict[str, float] = {}

def _deal_cache_get(key: str):
    ts = _deal_timestamps.get(key)
    if ts is None:
        return None
    if (monotonic() - ts) > DEAL_TTL_HOURS * 3600:
        deal_cache.pop(key, None)
        _deal_timestamps.pop(key, None)
        return None
    return deal_cache.get(key)

def _deal_cache_set(key: str, value):
    deal_cache[key] = value
    _deal_timestamps[key] = monotonic()

# Улучшенный выбор изображения
import math

def _score_image_url(url: str) -> int:
    url_l = url.lower()
    score = 0
    # Приоритет форматов
    if url_l.endswith('.webp'):
        score += 3
    elif url_l.endswith('.jpg') or url_l.endswith('.jpeg'):
        score += 2
    elif url_l.endswith('.png'):
        score += 1
    # Эвристика: большие числа в имени файла — возможно, большой размер
    for hint in ['1080', '1200', '1440', '1600', '1920', '2048']:
        if hint in url_l:
            score += 2
    return score

def _head_ok(url: str) -> bool:
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        ct = r.headers.get('content-type', '')
        return r.status_code == 200 and ('image/' in ct)
    except Exception:
        return False

def get_image_url(entry) -> str | None:
    candidates: list[str] = []
    try:
        media = getattr(entry, 'media_content', None)
        if media and isinstance(media, list):
            for m in media:
                u = m.get('url') if isinstance(m, dict) else None
                if u:
                    candidates.append(u)
        thumbs = getattr(entry, 'media_thumbnail', None)
        if thumbs and isinstance(thumbs, list):
            for m in thumbs:
                u = m.get('url') if isinstance(m, dict) else None
                if u:
                    candidates.append(u)
        enclosures = getattr(entry, 'enclosures', None)
        if enclosures and isinstance(enclosures, list):
            for enc in enclosures:
                u = enc.get('href') if isinstance(enc, dict) else None
                if u:
                    candidates.append(u)
        content = entry.get('content', []) if isinstance(entry, dict) else []
        for c in content:
            val = c.get('value', '')
            if 'img ' in val:
                import re as _re
                m = _re.search(r'src=["\']([^"\']+)', val)
                if m:
                    candidates.append(m.group(1))
    except Exception:
        pass

    # Сортируем кандидатов по эвристическому скору
    candidates = list(dict.fromkeys(candidates))  # уникальные
    candidates.sort(key=_score_image_url, reverse=True)

    # Проверяем HEAD у топ-3
    for u in candidates[:3]:
        if _head_ok(u):
            return u
    # Если HEAD не подтверждает, возвращаем лучший найденный
    return candidates[0] if candidates else None

# Список RSS-фидов
rss_feeds = [
    'https://www.gameinformer.com/rss',
    'https://www.playground.ru/rss/news.xml',
]

# Хранилище для последних новостей и кэш скидок
last_posts = {}
deal_cache = {}  # Кэш для результатов API скидок

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
    # Условные запросы по ETag/Last-Modified
    etag = rss_etags.get(url)
    last_mod = rss_last_modified.get(url)
    if etag:
        headers['If-None-Match'] = etag
    if last_mod:
        headers['If-Modified-Since'] = last_mod
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
            if resp.status_code == 200:
                # Сохраняем ETag/Last-Modified для будущих запросов
                et = resp.headers.get('ETag')
                lm = resp.headers.get('Last-Modified')
                if et:
                    rss_etags[url] = et
                if lm:
                    rss_last_modified[url] = lm
                return resp
            if resp.status_code == 304:
                logging.info(f'RSS не изменился (304): {url}')
                return None
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

def extract_game_title(title: str) -> str:
    """Извлекает название игры из заголовка новости."""
    title = title.strip()
    # Удаляем кавычки, если название в них
    if title.startswith('"') and title.endswith('"'):
        title = title[1:-1]
    # Берем до двоеточия или первые 4 слова
    if ':' in title:
        return title.split(':', 1)[0].strip()
    words = title.split()
    return ' '.join(words[:4]).strip()

def check_best_deal(game_title: str) -> str | None:
    """
    Проверяет скидки на игру через CheapShark API, возвращает лучшую.
    Кэширует результат, чтобы не запрашивать повторно.
    """
    # Проверяем кэш
    if game_title in deal_cache:
        return deal_cache[game_title]

    search_url = f"https://www.cheapshark.com/api/1.0/deals?storeID=1,2,3,7,11,15,23,25,29,30,31&title={game_title.replace(' ', '+')}&limit=10"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                deals = []
                for deal in data:
                    sale_price = float(deal['salePrice'])
                    normal_price = float(deal['normalPrice'])
                    if sale_price < normal_price:
                        discount = int((1 - sale_price / normal_price) * 100)
                        if discount >= MIN_DISCOUNT_PERCENT:
                            store_id = deal['storeID']
                            # Маппинг ID магазинов на названия
                            store_names = {
                                '1': 'Steam',
                                '2': 'GamersGate',
                                '3': 'GreenManGaming',
                                '7': 'GOG',
                                '11': 'Humble Bundle',
                                '15': 'Fanatical',
                                '23': 'GameBillet',
                                '25': 'Epic Games',
                                '29': '2game',
                                '30': 'IndieGala',
                                '31': 'Blizzard'
                            }
                            store_name = store_names.get(store_id, f'Store {store_id}')
                            deal_link = f"https://www.cheapshark.com/redirect?dealID={deal['dealID']}"
                            deals.append({
                                'discount': discount,
                                'store': store_name,
                                'price': sale_price,
                                'link': deal_link
                            })
                
                if deals:
                    # Сортируем по скидке (убывание)
                    best_deal = max(deals, key=itemgetter('discount'))
                    deal_text = f"🔥 Лучшая скидка {best_deal['discount']}% в {best_deal['store']} за ${best_deal['price']}! Купить: {best_deal['link']}"
                    deal_cache[game_title] = deal_text  # Кэшируем
                    return deal_text
    except Exception as e:
        logging.warning(f"Ошибка проверки скидок для '{game_title}': {e}")
    deal_cache[game_title] = None  # Кэшируем отсутствие скидок
    return None

def check_top_deals(game_title: str, limit: int | None = None) -> list[str]:
    """
    Возвращает список до limit лучших скидок с названиями магазинов и ссылками.
    Кэшируем результат по названию игры.
    """
    if limit is None:
        limit = MAX_DEALS
    cache_key = f"top:{game_title}:{limit}"
    cached = _deal_cache_get(cache_key)
    if cached is not None:
        return cached

    search_url = (
        "https://www.cheapshark.com/api/1.0/deals?"
        "storeID=1,2,3,7,11,15,23,25,29,30,31&"
        f"title={game_title.replace(' ', '+')}&limit=20"
    )
    deals_found: list[dict] = []
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json() or []
            store_names = {
                '1': 'Steam', '2': 'GamersGate', '3': 'GreenManGaming', '7': 'GOG',
                '11': 'Humble Bundle', '15': 'Fanatical', '23': 'GameBillet', '25': 'Epic Games',
                '29': '2game', '30': 'IndieGala', '31': 'Blizzard'
            }
            for deal in data:
                try:
                    sale_price = float(deal.get('salePrice', '0') or 0)
                    normal_price = float(deal.get('normalPrice', '0') or 0)
                    if sale_price <= 0 or normal_price <= 0 or sale_price >= normal_price:
                        continue
                    discount = int((1 - sale_price / normal_price) * 100)
                    if discount < MIN_DISCOUNT_PERCENT:
                        continue
                    store_id = str(deal.get('storeID', ''))
                    store_name = store_names.get(store_id, f'Store {store_id}')
                    deal_link = f"https://www.cheapshark.com/redirect?dealID={deal.get('dealID')}"
                    deals_found.append({
                        'discount': discount,
                        'store': store_name,
                        'price': sale_price,
                        'link': deal_link,
                    })
                except Exception:
                    continue
    except Exception as e:
        logging.warning(f"Ошибка проверки скидок для '{game_title}': {e}")

    deal_lines: list[str] = []
    if deals_found:
        deals_found.sort(key=lambda d: d['discount'], reverse=True)
        for d in deals_found[:limit]:
            deal_lines.append(
                f"💸 {d['store']}: −{d['discount']}% за ${d['price']} — <a href=\"{html_lib.escape(d['link'])}\">купить</a>"
            )

    _deal_cache_set(cache_key, deal_lines)
    return deal_lines

async def send_telegram_message(bot: telegram.Bot, chat_id: str, text: str):
    logging.info(f"Отправка сообщения в {chat_id}: {text[:50]}...")
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            disable_web_page_preview=False
        )
        logging.info("Сообщение успешно отправлено")
        await asyncio.sleep(2)  # Задержка 2 секунды
    except Exception as e:
        logging.error(f"Ошибка отправки: {e}")
        raise e

def process_news(title, summary, link):
    try:
        safe_link = normalize_url(link or '')
        message_html = f"<b>📰 {html_lib.escape(title)}</b>\n\nИсточник: {short_source_anchor(safe_link)}"
        
        # Клиент Hugging Face (по умолчанию отключён, если флаг не выставлен)
        client = None
        if HF_TOKEN and GENERATE_VIA_HF == '1':
            client = InferenceClient(token=HF_TOKEN)

        if client:
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
            
            message_html = f"<b>📰 {html_lib.escape(title)}</b>\n\n{html_lib.escape(summary_text)}\n\nИсточник: {short_source_anchor(safe_link)}"
        
        # Проверка скидок (теперь всегда пробуем добавить, если найдены)
        game_title = extract_game_title(title)
        deal_lines = check_top_deals(game_title, MAX_DEALS)
        if deal_lines:
            message_html += "\n\n" + "\n".join(deal_lines)

        return message_html
    except Exception as e:
        logging.error(f"Ошибка HF API: {e}")
        return f"<b>📰 {html_lib.escape(title)}</b>\n\nИсточник: {short_source_anchor(safe_link)}"

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
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = request_with_retries(feed_url)
                    if not resp:
                        logging.warning(f"Не удалось загрузить RSS-ленту {feed_url} после {MAX_RETRIES} попыток")
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
                                # Не дублируем заголовок: используем только сгенерированный текст + источник
                                message_html = (
                                    f"{html_lib.escape(generated)}\n\n"
                                    f"Источник: {short_source_anchor(safe_link)}"
                                )
                                # Добавляем скидки, если найдены
                                game_title = extract_game_title(title)
                                deal_lines = check_top_deals(game_title, MAX_DEALS)
                                if deal_lines:
                                    message_html += "\n\n" + "\n".join(deal_lines)
                            else:
                                # Базовый вариант без генерации: заголовок + источник
                                safe_link = normalize_url(link or '')
                                message_html = (
                                    f"<b>📰 {html_lib.escape(title)}</b>\n\n"
                                    f"Источник: {short_source_anchor(safe_link)}"
                                )
                                # Тоже пытаемся добавить скидки
                                game_title = extract_game_title(title)
                                deal_lines = check_top_deals(game_title, MAX_DEALS)
                                if deal_lines:
                                    message_html += "\n\n" + "\n".join(deal_lines)

                            if image_url:
                                caption = message_html
                                if len(caption) > MAX_CAPTION:
                                    caption = caption[:MAX_CAPTION - 1] + '…'
                                for send_attempt in range(1, MAX_RETRIES + 1):
                                    try:
                                        await bot.send_photo(
                                            chat_id=CHANNEL_ID,
                                            photo=image_url,
                                            caption=caption,
                                            parse_mode='HTML'
                                        )
                                        break
                                    except RetryAfter as e:
                                        delay = int(getattr(e, 'retry_after', 5))
                                        logging.warning(f'Telegram просит подождать {delay}с (FloodWait). Попытка {send_attempt}/{MAX_RETRIES}')
                                        await asyncio.sleep(delay)
                                    except TimedOut as e:
                                        logging.warning(f'Сетевая ошибка Telegram (TimedOut): {e}. Попытка {send_attempt}/{MAX_RETRIES}')
                                        await asyncio.sleep(BACKOFF_BASE_SEC ** send_attempt)
                                    except NetworkError as e:
                                        logging.warning(f'Сетевая ошибка Telegram: {e}. Попытка {send_attempt}/{MAX_RETRIES}')
                                        await asyncio.sleep(BACKOFF_BASE_SEC ** send_attempt)
                                    except Exception as e:
                                        logging.error(f'Ошибка отправки фото: {e}')
                                        break
                            else:
                                for part in split_message(message_html, MAX_TELEGRAM_MESSAGE):
                                    for send_attempt in range(1, MAX_RETRIES + 1):
                                        try:
                                            await bot.send_message(
                                                chat_id=CHANNEL_ID,
                                                text=part,
                                                parse_mode='HTML',
                                                disable_web_page_preview=False
                                            )
                                            break
                                        except RetryAfter as e:
                                            delay = int(getattr(e, 'retry_after', 5))
                                            logging.warning(f'Telegram просит подождать {delay}с (FloodWait). Попытка {send_attempt}/{MAX_RETRIES}')
                                            await asyncio.sleep(delay)
                                        except TimedOut as e:
                                            logging.warning(f'Сетевая ошибка Telegram (TimedOut): {e}. Попытка {send_attempt}/{MAX_RETRIES}')
                                            await asyncio.sleep(BACKOFF_BASE_SEC ** send_attempt)
                                        except NetworkError as e:
                                            logging.warning(f'Сетевая ошибка Telegram: {e}. Попытка {send_attempt}/{MAX_RETRIES}')
                                            await asyncio.sleep(BACKOFF_BASE_SEC ** send_attempt)
                                        except Exception as e:
                                            logging.error(f'Ошибка постинга: {e}')
                                            break

                            last_posts[post_key] = True
                            save_last_posts()
                            logging.info(f"Опубликовано: {title}")
                            news_sent += 1
                        except Exception as e:
                            logging.error(f'Ошибка обработки записи фида: {e}')
                    break
                except Exception as e:
                    logging.warning(f'Ошибка загрузки RSS {feed_url}: {e}. Попытка {attempt}/{MAX_RETRIES}')
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(BACKOFF_BASE_SEC ** attempt)
                    else:
                        logging.error(f'Не удалось загрузить RSS {feed_url} после {MAX_RETRIES} попыток')

def handle_signal(signum, frame):
    logging.info(f'Получен сигнал {signum}. Завершаем работу...')
    stop_event.set()

async def main():
    global last_posts
    last_posts = load_last_posts()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logging.info("Бот запущен. Проверяет новости каждые 6 часов.")

    schedule.clear()
    schedule.every(POST_INTERVAL_HOURS).hours.do(lambda: asyncio.create_task(fetch_and_post_news()))

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