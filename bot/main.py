"""Telegram bot for the open Nepali speech dataset (Phase 1: collection only).

Flow:
  /start -> Nepali consent summary + "I agree" button
  agree  -> POST /api/consent, then GET /api/prompts/next and send the prompt
  voice  -> download (Ogg/Opus) -> POST /api/recordings multipart, confirm,
            then send the next prompt
  voice before consent -> ask the user to run /start first
  /help  -> short Nepali/English help

The bot talks to the backend API over HTTP only (no direct database access).
Token comes from TELEGRAM_BOT_TOKEN env only; API base URL from API_BASE_URL
(default http://localhost:8000).
"""

import io
import logging
import os
import sys

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

CONSENT_VERSION = "v1"
DEFAULT_API_BASE_URL = "http://localhost:8000"

CONSENT_TEXT = (
    "नमस्ते! 🙏\n\n"
    "यो बट खुला नेपाली बोली डाटासेटका लागि आवाज सङ्कलन गर्छ।\n\n"
    "• तपाईंका रेकर्डिङहरू CC0 इजाजतपत्रमा सार्वजनिक गरिनेछन् — "
    "जसले पनि प्रयोग गर्न सक्नेछ।\n"
    "• तपाईंको रेकर्डिङसँग कुनै व्यक्तिगत जानकारी जोडिँदैन — "
    "नाम, फोन नम्बर वा ठेगाना सङ्कलन गरिँदैन।\n"
    "• तपाईं जुनसुकै बेला रोक्न सक्नुहुन्छ — कुनै बाध्यता छैन।\n\n"
    "सहमत हुन तलको बटन थिच्नुहोस्।"
)

HELP_TEXT = (
    "सहयोग / Help\n\n"
    "• /start — सहमति दिनुहोस् र रेकर्डिङ सुरु गर्नुहोस्\n"
    "• भ्वाइस सन्देश पठाउनुहोस् — देखाइएको वाक्य बोल्नुहोस्\n"
    "• /help — यो सहयोग सन्देश\n\n"
    "Send a voice message to record the shown sentence. "
    "You can stop any time."
)

SERVER_DOWN_TEXT = "सर्भर अहिले उपलब्ध छैन, कृपया पछि प्रयास गर्नुहोस्।"
NEED_START_TEXT = "कृपया पहिले /start गरेर सहमति दिनुहोस्।"
AGREE_BUTTON_TEXT = "✅ म सहमत छु"
CONSENT_CALLBACK_DATA = "consent_agree"


# ---------------------------------------------------------------------------
# Pure helpers (covered by bot/test_bot.py, no network / no token needed)
# ---------------------------------------------------------------------------

def contributor_id(user_id: int) -> str:
    """Contributor id for a Telegram user, per the project convention."""
    return f"tg:{user_id}"


def build_consent_payload(contributor: str, version: str = CONSENT_VERSION) -> dict:
    """JSON body for POST /api/consent. Metadata fields are optional."""
    return {
        "contributor_id": contributor,
        "version": version,
        "province": None,
        "age_range": None,
        "gender": None,
    }


def format_prompt_message(prompt: dict) -> str:
    """Message shown for one prompt. ``prompt`` is the /api/prompts/next body."""
    if prompt.get("done") or prompt.get("prompt_id") is None:
        total = prompt.get("total", "")
        return f"🎉 बधाई छ! सबै {total} वाक्यहरू रेकर्ड गरियो। धन्यवाद! 🙏"
    text = prompt["text"]
    domain = prompt.get("domain") or ""
    line = f"कृपया यो वाक्य बोल्नुहोस्:\n\n“{text}”"
    if domain:
        line += f"\n\nविषय: {domain}"
    line += "\n\n🎙️ भ्वाइस सन्देश पठाउनुहोस्।"
    return line


def format_done_message(total: int | None = None) -> str:
    """Message when the contributor has recorded every prompt."""
    if total:
        return f"🎉 बधाई छ! सबै {total} वाक्यहरू रेकर्ड गरियो। धन्यवाद! 🙏"
    return "🎉 बधाई छ! सबै वाक्यहरू रेकर्ड गरियो। धन्यवाद! 🙏"


# ---------------------------------------------------------------------------
# API client (httpx; a client/transport can be injected for tests)
# ---------------------------------------------------------------------------

def get_api_base_url() -> str:
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def post_consent(base_url: str, contributor: str,
                 client: httpx.Client | None = None) -> dict:
    """POST /api/consent. Returns the parsed JSON body. Raises on HTTP error."""
    own = client is None
    client = client or httpx.Client(base_url=base_url, timeout=15.0)
    try:
        resp = client.post("/api/consent", json=build_consent_payload(contributor))
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            client.close()


def fetch_next_prompt(base_url: str, contributor: str,
                      client: httpx.Client | None = None) -> dict:
    """GET /api/prompts/next. Returns the parsed JSON body."""
    own = client is None
    client = client or httpx.Client(base_url=base_url, timeout=15.0)
    try:
        resp = client.get("/api/prompts/next",
                          params={"contributor_id": contributor})
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            client.close()


def upload_recording(base_url: str, prompt_id: int, contributor: str,
                     audio_bytes: bytes, filename: str = "voice.ogg",
                     client: httpx.Client | None = None) -> dict:
    """POST /api/recordings (multipart). Returns the parsed JSON body."""
    own = client is None
    client = client or httpx.Client(base_url=base_url, timeout=60.0)
    try:
        files = {"audio": (filename, io.BytesIO(audio_bytes), "audio/ogg")}
        data = {"prompt_id": str(prompt_id), "contributor_id": contributor}
        resp = client.post("/api/recordings", data=data, files=files)
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            client.close()


# ---------------------------------------------------------------------------
# Bot state + handlers
# ---------------------------------------------------------------------------

# In-memory set of Telegram user ids that have consented in this run.
# (Consent itself is stored server-side; this only gates voice handling.
#  A 403 "consent required" from the API is also treated as not-consented.)
consented_users: set[int] = {}


async def send_next_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           contributor: str) -> None:
    """Fetch the next prompt for this contributor and send it."""
    try:
        prompt = fetch_next_prompt(get_api_base_url(), contributor)
    except httpx.HTTPError:
        logger.exception("API error fetching next prompt")
        await update.effective_message.reply_text(SERVER_DOWN_TEXT)
        return
    _track_prompt_sent(context, prompt)
    await update.effective_message.reply_text(format_prompt_message(prompt))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[InlineKeyboardButton(AGREE_BUTTON_TEXT,
                                      callback_data=CONSENT_CALLBACK_DATA)]]
    await update.message.reply_text(
        CONSENT_TEXT, reply_markup=InlineKeyboardMarkup(keyboard))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)


async def on_consent_agree(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    contributor = contributor_id(user_id)
    try:
        post_consent(get_api_base_url(), contributor)
    except httpx.HTTPError:
        logger.exception("API error posting consent")
        await query.edit_message_text(SERVER_DOWN_TEXT)
        return
    consented_users.add(user_id)
    await query.edit_message_text("धन्यवाद! तपाईंको सहमति दर्ता भयो। 🙏")
    # Send the first prompt as a fresh message.
    try:
        prompt = fetch_next_prompt(get_api_base_url(), contributor)
    except httpx.HTTPError:
        logger.exception("API error fetching first prompt")
        await query.message.reply_text(SERVER_DOWN_TEXT)
        return
    _track_prompt_sent(context, prompt)
    await query.message.reply_text(format_prompt_message(prompt))


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = message.from_user.id
    contributor = contributor_id(user_id)

    if user_id not in consented_users:
        await message.reply_text(NEED_START_TEXT)
        return

    # The prompt this voice message answers is the one we last sent.
    prompt_id = context.user_data.get("current_prompt_id")
    if prompt_id is None:
        # No prompt in flight (e.g. bot restarted): fetch the next one and
        # ask the user to record that instead of guessing.
        await message.reply_text(
            "कृपया पहिले /start गरेर नयाँ वाक्य लिनुहोस्।")
        return

    try:
        voice_file = await message.voice.get_file()
        buf = io.BytesIO()
        await voice_file.download_to_memory(buf)
        audio_bytes = buf.getvalue()
    except Exception:
        logger.exception("Failed to download voice message")
        await message.reply_text(
            "भ्वाइस डाउनलोड गर्न सकिएन, कृपया फेरि प्रयास गर्नुहोस्।")
        return

    try:
        upload_recording(get_api_base_url(), prompt_id, contributor, audio_bytes)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            consented_users.discard(user_id)
            await message.reply_text(NEED_START_TEXT)
        elif exc.response.status_code == 413:
            await message.reply_text(
                "रेकर्डिङ धेरै ठूलो छ (५ MB भन्दा बढी), कृपया छोटो रेकर्ड गर्नुहोस्।")
        else:
            logger.exception("API error uploading recording")
            await message.reply_text(SERVER_DOWN_TEXT)
        return
    except httpx.HTTPError:
        logger.exception("API error uploading recording")
        await message.reply_text(SERVER_DOWN_TEXT)
        return

    context.user_data.pop("current_prompt_id", None)
    await message.reply_text("✅ रेकर्डिङ प्राप्त भयो, धन्यवाद!")
    await send_next_prompt(update, context, contributor)


def _track_prompt_sent(context: ContextTypes.DEFAULT_TYPE, prompt: dict) -> None:
    """Remember which prompt id the last sent prompt was, for voice upload."""
    if prompt.get("prompt_id") is not None:
        context.user_data["current_prompt_id"] = prompt["prompt_id"]


def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(on_consent_agree,
                                         pattern=f"^{CONSENT_CALLBACK_DATA}$"))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    return app


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print(
            "ERROR: TELEGRAM_BOT_TOKEN environment variable is not set. "
            "Create a bot with @BotFather, copy its token, and set "
            "TELEGRAM_BOT_TOKEN before running the bot.",
            file=sys.stderr,
        )
        return 1
    app = build_application(token)
    logger.info("Starting Nepali speech collection bot (API: %s)",
                get_api_base_url())
    app.run_polling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
