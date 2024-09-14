import json
import requests
from abc import ABC, abstractmethod
from config import AI_MODELS

class AIModel(ABC):
    def __init__(self, api_key, model_name=None):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    async def generate_text(self, prompt):
        pass

class OpenAIModel(AIModel):
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.api_base = "https://api.openai.com/v1"

    async def generate_text(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Failed to generate text from OpenAI API: {response.text}")

class OpenRouterModel(AIModel):
    def __init__(self, api_key, model_name="openai/gpt-3.5-turbo"):
        super().__init__(api_key, model_name)

    async def generate_text(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

class CloudflareAIModel(AIModel):
    def __init__(self, api_key, model_name="@cf/meta/llama-2-7b-chat-int8"):
        super().__init__(api_key, model_name)

    async def generate_text(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{self.api_key}/ai/run/{self.model_name}", headers=headers, json=data)
        return response.json()['result']['response']

class GoogleGeminiModel(AIModel):
    def __init__(self, api_key, model_name="gemini-1.5-flash-001"):
        super().__init__(api_key, model_name)

    async def generate_text(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.9,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "NONE"
                }
            ]
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent",
            headers=headers,
            json=data
        )
        response_json = response.json()
        if 'candidates' in response_json and response_json['candidates']:
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception("Failed to generate text from Google Gemini API")

class CustomOpenAIModel(OpenAIModel):
    def __init__(self, api_key, endpoint, model_name="gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.api_base = endpoint

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.load_config()

    def set_model(self, model_key: str) -> bool:
        if model_key not in AI_MODELS:
            return False
        
        model_config = AI_MODELS[model_key]
        api_key = model_config["api_key"]
        model_name = model_config.get("model_name")
        endpoint = model_config.get("endpoint")

        if not api_key:
            return False

        if model_key == "openai":
            self.models[model_key] = OpenAIModel(api_key, model_name)
        elif model_key == "openrouter":
            self.models[model_key] = OpenRouterModel(api_key, model_name)
        elif model_key == "cloudflare":
            self.models[model_key] = CloudflareAIModel(api_key, model_name)
        elif model_key == "google":
            self.models[model_key] = GoogleGeminiModel(api_key, model_name)
        elif model_key == "custom_openai":
            self.models[model_key] = CustomOpenAIModel(api_key, endpoint, model_name)
        else:
            return False
        
        self.current_model = model_key
        self.save_config()
        return True

    async def generate_text(self, prompt: str) -> str:
        if self.current_model and self.current_model in self.models:
            return await self.models[self.current_model].generate_text(prompt)
        raise ValueError("No model selected or model not found")

    def get_current_model(self) -> dict:
        if self.current_model:
            model = self.models[self.current_model]
            return {
                "name": self.current_model,
                "api_key": model.api_key,
                "model_name": model.model_name
            }
        return None

    def get_model_list(self) -> list:
        return list(AI_MODELS.keys())

    def save_config(self):
        config = {
            "current_model": self.current_model
        }
        with open('model_config.json', 'w') as f:
            json.dump(config, f)

    def load_config(self):
        try:
            with open('model_config.json', 'r') as f:
                config = json.load(f)
                self.current_model = config.get("current_model")
                if self.current_model:
                    self.set_model(self.current_model)
        except FileNotFoundError:
            # 如果文件不存在，使用默认配置
            pass

__all__ = [
    'AIModel',
    'ModelManager',
    'OpenAIModel',
    'OpenRouterModel',
    'CloudflareAIModel',
    'GoogleGeminiModel',
    'CustomOpenAIModel'
]

# combined_config.py
# 这个文件包含了config目录下所有Python文件的内容

import os
import logging

# ----------------------
# ai_model_config.py
# ----------------------

# AI Model API Keys (初始化为空字符串，稍后通过聊天窗口设置)
OPENAI_API_KEY = ""
OPENROUTER_API_KEY = ""
CLOUDFLARE_API_KEY = ""
GEMINI_API_KEY = ""
CUSTOM_OPENAI_API_KEY = ""
CUSTOM_OPENAI_ENDPOINT = ""

# AI模型配置
AI_MODELS = {
    "openai": {"name": "OpenAI", "api_key": OPENAI_API_KEY},
    "openrouter": {"name": "OpenRouter", "api_key": OPENROUTER_API_KEY},
    "cloudflare": {"name": "Cloudflare AI", "api_key": CLOUDFLARE_API_KEY},
    "google": {"name": "Google Gemini", "api_key": GEMINI_API_KEY},
    "custom_openai": {"name": "Custom OpenAI Endpoint", "api_key": CUSTOM_OPENAI_API_KEY, "endpoint": CUSTOM_OPENAI_ENDPOINT}
}

def update_api_key(model_name, api_key):
    global AI_MODELS, OPENAI_API_KEY, OPENROUTER_API_KEY, CLOUDFLARE_API_KEY, GEMINI_API_KEY, CUSTOM_OPENAI_API_KEY
    if model_name in AI_MODELS:
        AI_MODELS[model_name]["api_key"] = api_key
        if model_name == "openai":
            OPENAI_API_KEY = api_key
        elif model_name == "openrouter":
            OPENROUTER_API_KEY = api_key
        elif model_name == "cloudflare":
            CLOUDFLARE_API_KEY = api_key
        elif model_name == "google":
            GEMINI_API_KEY = api_key
        elif model_name == "custom_openai":
            CUSTOM_OPENAI_API_KEY = api_key
        return True
    return False

def update_custom_openai_endpoint(endpoint):
    global AI_MODELS, CUSTOM_OPENAI_ENDPOINT
    AI_MODELS["custom_openai"]["endpoint"] = endpoint
    CUSTOM_OPENAI_ENDPOINT = endpoint

# ----------------------
# bot_config.py
# ----------------------

# Telegram Bot Token
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Webhook URL
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

# 控制面板管理员ID列表
PANEL_ADMINS = [int(id) for id in os.getenv('PANEL_ADMINS', '').split(',') if id]

# 管理员命令
ADMIN_COMMANDS = ['/toggle', '/ban', '/unban', '/warn', '/set_api_key']

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = 'bot.log'

# 数据存储配置
DATA_FILE = 'user_data.json'

# 性能配置
CACHE_SIZE = 1000  # 缓存大小
RATE_LIMIT = 5  # 每秒最大请求数

# 隐私设置
PRIVACY_MODE = True  # 是否启用隐私模式

# 语言设置
LANGUAGE = 'zh-CN'

# 时区设置
TIMEZONE = 'Asia/Shanghai'

# 错误通知配置
ERROR_NOTIFICATION = {
    'enabled': True,  # 是否启用错误通知
    'admin_chat_id': int(os.getenv('ADMIN_CHAT_ID', 0)),  # 管理员聊天ID，用于接收错误通知
}

# 创建BOT_CONFIG字典
BOT_CONFIG = {
    'TOKEN': TOKEN,
    'WEBHOOK_URL': WEBHOOK_URL,
    'PANEL_ADMINS': PANEL_ADMINS,
    'ADMIN_COMMANDS': ADMIN_COMMANDS,
    'LOG_LEVEL': LOG_LEVEL,
    'LOG_FILE': LOG_FILE,
    'DATA_FILE': DATA_FILE,
    'CACHE_SIZE': CACHE_SIZE,
    'RATE_LIMIT': RATE_LIMIT,
    'PRIVACY_MODE': PRIVACY_MODE,
    'LANGUAGE': LANGUAGE,
    'TIMEZONE': TIMEZONE,
    'ERROR_NOTIFICATION': ERROR_NOTIFICATION
}

# ----------------------
# feature_config.py
# ----------------------

# 功能开关
FEATURES = {
    'spam_detection': True,
    'auto_ban': True,
    'welcome_message': True,
    'user_stats': True,
    'warning_system': True,
    'ignore_admins': True,
    'silent_mode': False,  # 新添加的静音模式开关
}

# ----------------------
# logging_config.py
# ----------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")

# ----------------------
# spam_detection_config.py
# ----------------------

# 垃圾检测配置
SPAM_DETECTION = {
    'min_chars': 5,  # 最小字符数
    'max_emojis': 5,  # 最大emoji数
    'max_urls': 2,  # 最大URL数
    'banned_words': ['广告', '推广', '优惠', '折扣', '限时', '抢购', '秒杀', '免费'],  # 禁用词列表
    'sensitivity': 0.7,  # 敏感度（0-1）
}

# ----------------------
# warning_ban_config.py
# ----------------------

# 警告系统配置
WARNING_SYSTEM = {
    'max_warnings': 3,  # 最大警告次数
    'warning_expiry_days': 30,  # 警告过期天数
}

# 自动封禁配置
AUTO_BAN = {
    'threshold': 0.9,  # 自动封禁阈值（0-1）
    'ban_duration_hours': 24,  # 封禁时长（小时）
}

# 创建WARNING_BAN_CONFIG
WARNING_BAN_CONFIG = {
    'WARNING_SYSTEM': WARNING_SYSTEM,
    'AUTO_BAN': AUTO_BAN
}

# ----------------------
# __init__.py
# ----------------------

__all__ = [
    'AI_MODELS',
    'update_api_key',
    'update_custom_openai_endpoint',
    'BOT_CONFIG',
    'PANEL_ADMINS',
    'ADMIN_COMMANDS',
    'FEATURES',
    'setup_logging',
    'SPAM_DETECTION',
    'WARNING_BAN_CONFIG',
    'ERROR_NOTIFICATION'
]

import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from spam_detector.spam_detector import SpamDetector
from user_management.user_manager import UserManager
from ai_models.model_manager import ModelManager
from config.feature_config import FEATURES
from handlers.command_handlers import start, help, show_control_panel, warn_user, set_api_key
from handlers.message_handlers import handle_message, welcome_new_member, update_chat_admins
from utils.error_handler import error_handler
from telegram import Update
from telegram.ext import ContextTypes, CallbackQueryHandler
from config import FEATURES, AI_MODELS
from handlers.menu_handlers import MenuHandlers
from handlers.action_handlers import ActionHandlers
from handlers.input_handlers import InputHandlers
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
import os

logger = logging.getLogger(__name__)

# SpamSentinelBot class
class SpamSentinelBot:
    def __init__(self, application: Application):
        self.application = application
        self.spam_detector = None
        self.user_manager = None
        self.model_manager = None
        self.features = FEATURES
        self.managed_chats = set()
        self.paused_chats = {}
        self.control_panel = None
        logger.info("哨兵机器人已初始化")

    async def initialize(self):
        logger.info("正在初始化哨兵机器人组件")
        self.spam_detector = await SpamDetector.create()
        self.user_manager = await UserManager.create()
        self.model_manager = ModelManager()
        self.control_panel = ControlPanel(self)
        await self.control_panel.setup_handlers()
        await self.setup_handlers()
        
        self.application.bot_data['paused_chats'] = self.paused_chats
        
        logger.info("哨兵机器人初始化完成")

    async def setup_handlers(self):
        logger.info("正在设置消息处理程序")
        self.application.add_handler(CommandHandler("start", start))
        self.application.add_handler(CommandHandler("help", help))
        self.application.add_handler(CommandHandler("control", show_control_panel))
        self.application.add_handler(CommandHandler("warn", warn_user))
        self.application.add_handler(CommandHandler("set_api_key", set_api_key))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_member))
        self.application.add_handler(MessageHandler(filters.ChatType.GROUPS, update_chat_admins))
        self.application.add_error_handler(error_handler)
        logger.info("消息处理程序设置完成")

    async def toggle_feature(self, feature):
        if feature in self.features:
            self.features[feature] = not self.features[feature]
            return self.features[feature]
        return None

# Application initialization
application = None
bot = None

async def initialize_application():
    global application, bot
    if application is None:
        logger.info("正在初始化应用程序")
        token = os.environ.get("BOT_TOKEN")
        if token is None:
            logger.error("未设置 BOT_TOKEN 环境变量")
            raise ValueError("未设置 BOT_TOKEN 环境变量")
        application = Application.builder().token(token).build()
        bot = SpamSentinelBot(application)
        logger.info("应用程序初始化成功")

def get_application():
    global application
    return application

def get_bot():
    global bot
    return bot

# ControlPanel class
class ControlPanel:
    def __init__(self, bot):
        self.bot = bot
        self.menu_handlers = MenuHandlers(self)
        self.action_handlers = ActionHandlers(self)
        self.input_handlers = InputHandlers(self)

    async def setup_handlers(self):
        self.bot.application.add_handler(CallbackQueryHandler(self.button_callback))

    async def show_control_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.bot.user_manager.is_panel_admin(update.effective_user.id):
            await update.message.reply_text("抱歉，只有控制面板管理员可以访问控制面板。")
            return

        await update.message.reply_text("欢迎使用 SpamSentinel 控制面板。请选择一个选项：", 
                                        reply_markup=self.menu_handlers.get_main_menu())

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if not self.bot.user_manager.is_panel_admin(query.from_user.id):
            await query.edit_message_text("抱歉，只有控制面板管理员可以使用控制面板。")
            return

        action = query.data.split('_')[0]
        if action == 'menu':
            await self.menu_handlers.handle_menu(query)
        elif action == 'toggle':
            await self.action_handlers.handle_toggle(query)
        elif action == 'model':
            await self.action_handlers.handle_model_selection(query, context)
        elif action == 'action':
            await self.action_handlers.handle_action(query, context)

    async def process_user_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.input_handlers.process_user_input(update, context)

    def toggle_feature(self, feature):
        FEATURES[feature] = not FEATURES[feature]
        return FEATURES[feature]

# FastAPI routes
router = APIRouter()

@router.post('/webhook/{token}')
async def webhook(token: str, request: Request):
    logger.info(f"Received webhook request for token: {token[:5]}...")
    application = get_application()
    bot = get_bot()
    
    if application and application.bot.token == token:
        logger.info("Token verified, processing update")
        try:
            update_data = await request.json()
            logger.debug(f"Received update data: {update_data}")
            update = Update.de_json(update_data, application.bot)
            await bot.application.process_update(update)
            logger.info("Update processed successfully")
            return Response(content='OK', status_code=200)
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")
            return Response(content='Internal Server Error', status_code=500)
    else:
        logger.warning("Unauthorized webhook access attempt")
        return Response(content='Unauthorized', status_code=401)

@router.get('/', response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Received request for home page")
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SpamSentinel Bot</title>
    </head>
    <body>
        <h1>SpamSentinel Bot</h1>
        <p>Bot is running. Use Telegram to interact with it.</p>
    </body>
    </html>
    '''
    return HTMLResponse(content=html)

# merged_handlers.py
# 这个文件是由以下文件合并而成：
# __init__.py, action_handlers.py, command_handlers.py, input_handlers.py, menu_handlers.py, message_handlers.py

import logging
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from config import PANEL_ADMINS, AI_MODELS, FEATURES
from utils import notify_admin, update_custom_openai_endpoint, update_api_key

logger = logging.getLogger(__name__)

# 从 __init__.py
__all__ = [
    'ActionHandlers',
    'start',
    'help',
    'show_control_panel',
    'warn_user',
    'set_api_key',
    'InputHandlers',
    'MenuHandlers',
    'handle_message',
    'welcome_new_member',
    'update_chat_admins',
    'handle_private_message'
]

# 从 action_handlers.py
class ActionHandlers:
    def __init__(self, control_panel):
        self.control_panel = control_panel

    async def handle_button(self, query, context):
        action = query.data.split('_')[0]
        if action == 'toggle':
            await self.handle_toggle(query)
        elif action == 'model':
            await self.handle_model_selection(query, context)
        elif action == 'action':
            await self.handle_action(query, context)
        else:
            await query.answer("未知的按钮操作")

    async def handle_toggle(self, query):
        feature = query.data.replace('toggle_', '')
        if feature == 'log_level':
            current_level = logging.getLogger().level
            new_level = logging.DEBUG if current_level != logging.DEBUG else logging.INFO
            logging.getLogger().setLevel(new_level)
            await query.edit_message_text(f"日志级别已切换为: {'DEBUG' if new_level == logging.DEBUG else 'INFO'}", 
                                          reply_markup=self.control_panel.menu_handlers.get_system_settings_menu())
        else:
            new_state = self.control_panel.toggle_feature(feature)
            await query.edit_message_text(f"{feature.replace('_', ' ').title()} 现在已{'开启' if new_state else '关闭'}。", 
                                          reply_markup=self.control_panel.menu_handlers.get_toggles_menu())

    async def handle_model_selection(self, query, context):
        model_key = query.data.split('_')[1]
        context.user_data['selected_model'] = model_key
        await query.edit_message_text(f"已选择 {AI_MODELS[model_key]['name']} 模型。请输入API密钥：")
        context.user_data['awaiting_api_key'] = True

    async def handle_action(self, query, context):
        action = query.data.split('_')[1]
        if action == 'ban_user':
            await query.edit_message_text("请输入要封禁的用户 ID：")
            context.user_data['awaiting_ban_user_id'] = True
        elif action == 'unban_user':
            await query.edit_message_text("请输入要解封的用户 ID：")
            context.user_data['awaiting_unban_user_id'] = True
        elif action == 'add_panel_admin':
            await query.edit_message_text("请输入要添加为控制面板管理员的用户 ID：")
            context.user_data['awaiting_add_panel_admin_id'] = True
        elif action == 'remove_panel_admin':
            await query.edit_message_text("请输入要移除控制面板管理员权限的用户 ID：")
            context.user_data['awaiting_remove_panel_admin_id'] = True
        elif action == 'update_admins':
            await self.control_panel.bot.update_chat_admins(query, context)
            await query.edit_message_text("群组管理员列表已更新。", 
                                          reply_markup=self.control_panel.menu_handlers.get_system_settings_menu())

# 从 command_handlers.py
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received /start command from user {update.effective_user.id}")
    chat_id = update.effective_chat.id
    if update.effective_chat.type in ['group', 'supergroup']:
        if chat_id not in context.bot_data['managed_chats']:
            context.bot_data['managed_chats'].add(chat_id)
            await update_chat_admins(update, context)
        await update.message.reply_text("SpamSentinel 机器人已在此群组中启动。使用 /help 查看可用命令。")
    else:
        if update.effective_user.id in PANEL_ADMINS:
            await update.message.reply_text("欢迎使用 SpamSentinel 机器人控制面板。使用 /control 来访问控制面板。")
        else:
            await update.message.reply_text("抱歉，只有管理员才能私聊使用机器人。")

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received /help command from user {update.effective_user.id}")
    if update.effective_chat.type in ['group', 'supergroup']:
        help_text = """
        群组可用命令：
        /start - 在群组中启动机器人
        /help - 显示帮助信息
        /warn - 警告用户（仅管理员可用）
        """
    elif update.effective_user.id in PANEL_ADMINS:
        help_text = """
        管理员私聊可用命令：
        /start - 显示欢迎信息
        /help - 显示帮助信息
        /control - 显示控制面板
        /set_api_key - 设置AI模型API密钥

        您还可以发送任何消息来测试垃圾检测功能。
        """
    else:
        help_text = "抱歉，只有管理员才能私聊使用机器人。"
    await update.message.reply_text(help_text)

async def show_control_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received /control command from user {update.effective_user.id}")
    if update.effective_user.id in PANEL_ADMINS:
        if context.bot.control_panel:
            await context.bot.control_panel.show_control_panel(update, context)
        else:
            await update.message.reply_text("控制面板尚未初始化，请稍后再试。")
    else:
        await update.message.reply_text("抱歉，只有管理员才能访问控制面板。")

async def warn_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received /warn command from user {update.effective_user.id}")
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    
    if not await context.bot.user_manager.is_admin(chat_id, user_id):
        await update.message.reply_text("只有管理员可以使用此命令。")
        return

    if not context.args or not context.args[0].startswith('@'):
        await update.message.reply_text("请使用正确的格式：/warn @用户名")
        return

    target_username = context.args[0][1:]  # 去掉@
    target_user = await context.bot.get_chat_member(chat_id, target_username)
    
    if not target_user:
        await update.message.reply_text(f"找不到用户 @{target_username}")
        return

    warning = await context.bot.user_manager.warn_user(chat_id, target_user.user.id)
    
    if not context.bot.features['silent_mode']:
        await update.message.reply_text(f"已警告用户 {target_user.user.mention_html()}。这是第 {warning} 次警告。", parse_mode='HTML')

async def set_api_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received /set_api_key command from user {update.effective_user.id}")
    user_id = update.effective_user.id
    if user_id not in PANEL_ADMINS:
        await update.message.reply_text("抱歉，只有管理员才能设置API密钥。")
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text("请使用正确的格式：/set_api_key <model_name> <api_key>")
        return

    model_name = context.args[0].lower()
    api_key = context.args[1]

    if model_name not in AI_MODELS:
        await update.message.reply_text(f"未知的模型名称。可用的模型有：{', '.join(AI_MODELS.keys())}")
        return

    if model_name == "custom_openai" and len(context.args) == 3:
        endpoint = context.args[2]
        await update_custom_openai_endpoint(endpoint)
        await update.message.reply_text(f"已更新 Custom OpenAI Endpoint: {endpoint}")

    if await update_api_key(model_name, api_key):
        await update.message.reply_text(f"已成功更新 {model_name} 的API密钥。")
        await context.bot.model_manager.set_model(model_name)
    else:
        await update.message.reply_text("更新API密钥失败。请检查模型名称是否正确。")

# 从 input_handlers.py
class InputHandlers:
    def __init__(self, control_panel):
        self.control_panel = control_panel

    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self.control_panel.bot.user_manager.is_panel_admin(user_id):
            await update.message.reply_text("抱歉，只有控制面板管理员可以使用这些功能。")
            return

        if 'awaiting_api_key' in context.user_data and context.user_data['awaiting_api_key']:
            await self.process_api_key(update, context)
        elif 'awaiting_ban_user_id' in context.user_data:
            await self.process_ban_user(update, context)
        elif 'awaiting_unban_user_id' in context.user_data:
            await self.process_unban_user(update, context)
        elif 'awaiting_add_panel_admin_id' in context.user_data:
            await self.process_add_panel_admin(update, context)
        elif 'awaiting_remove_panel_admin_id' in context.user_data:
            await self.process_remove_panel_admin(update, context)

    async def process_api_key(self, update, context):
        model_key = context.user_data['selected_model']
        api_key = update.message.text
        del context.user_data['awaiting_api_key']
        success = self.control_panel.bot.model_manager.set_model(model_key, api_key)
        if success:
            await update.message.reply_text(f"{AI_MODELS[model_key]['name']} 模型设置成功。",
                                            reply_markup=self.control_panel.menu_handlers.get_ai_models_menu())
        else:
            await update.message.reply_text(f"设置 {AI_MODELS[model_key]['name']} 模型失败，请重试。",
                                            reply_markup=self.control_panel.menu_handlers.get_ai_models_menu())

    async def process_ban_user(self, update, context):
        try:
            user_id = int(update.message.text)
            await self.control_panel.bot.user_manager.ban_user(update.effective_chat.id, user_id)
            await update.message.reply_text(f"用户 {user_id} 已被封禁。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        except ValueError:
            await update.message.reply_text("无效的用户 ID。请输入一个有效的数字 ID。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        del context.user_data['awaiting_ban_user_id']

    async def process_unban_user(self, update, context):
        try:
            user_id = int(update.message.text)
            await self.control_panel.bot.user_manager.unban_user(update.effective_chat.id, user_id)
            await update.message.reply_text(f"用户 {user_id} 已被解封。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        except ValueError:
            await update.message.reply_text("无效的用户 ID。请输入一个有效的数字 ID。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        del context.user_data['awaiting_unban_user_id']

    async def process_add_panel_admin(self, update, context):
        try:
            user_id = int(update.message.text)
            self.control_panel.bot.user_manager.add_panel_admin(user_id)
            await update.message.reply_text(f"用户 {user_id} 已被添加为控制面板管理员。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        except ValueError:
            await update.message.reply_text("无效的用户 ID。请输入一个有效的数字 ID。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        del context.user_data['awaiting_add_panel_admin_id']

    async def process_remove_panel_admin(self, update, context):
        try:
            user_id = int(update.message.text)
            self.control_panel.bot.user_manager.remove_panel_admin(user_id)
            await update.message.reply_text(f"用户 {user_id} 的控制面板管理员权限已被移除。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        except ValueError:
            await update.message.reply_text("无效的用户 ID。请输入一个有效的数字 ID。", 
                                            reply_markup=self.control_panel.menu_handlers.get_user_management_menu())
        del context.user_data['awaiting_remove_panel_admin_id']

# 从 menu_handlers.py
class MenuHandlers:
    def __init__(self, control_panel):
        self.control_panel = control_panel

    async def show_main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("请选择一个选项：", reply_markup=self.get_main_menu())

    def get_main_menu(self):
        keyboard = [
            [InlineKeyboardButton("功能开关", callback_data='menu_toggles')],
            [InlineKeyboardButton("用户管理", callback_data='menu_user_management')],
            [InlineKeyboardButton("AI模型设置", callback_data='menu_ai_models')],
            [InlineKeyboardButton("系统设置", callback_data='menu_system_settings')],
            [InlineKeyboardButton("统计信息", callback_data='menu_stats')]
        ]
        return InlineKeyboardMarkup(keyboard)

    def get_toggles_menu(self):
        keyboard = [
            [
                InlineKeyboardButton(f"垃圾检测: {'开' if FEATURES['spam_detection'] else '关'}", 
                                     callback_data='toggle_spam_detection'),
                InlineKeyboardButton(f"自动封禁: {'开' if FEATURES['auto_ban'] else '关'}", 
                                     callback_data='toggle_auto_ban')
            ],
            [
                InlineKeyboardButton(f"欢迎消息: {'开' if FEATURES['welcome_message'] else '关'}", 
                                     callback_data='toggle_welcome_message'),
                InlineKeyboardButton(f"用户统计: {'开' if FEATURES['user_stats'] else '关'}", 
                                     callback_data='toggle_user_stats')
            ],
            [
                InlineKeyboardButton(f"警告系统: {'开' if FEATURES['warning_system'] else '关'}", 
                                     callback_data='toggle_warning_system'),
                InlineKeyboardButton(f"忽略管理员: {'开' if FEATURES['ignore_admins'] else '关'}", 
                                     callback_data='toggle_ignore_admins')
            ],
            [
                InlineKeyboardButton(f"静音模式: {'开' if FEATURES['silent_mode'] else '关'}", 
                                     callback_data='toggle_silent_mode')
            ],
            [InlineKeyboardButton("返回主菜单", callback_data='menu_main')]
        ]
        return InlineKeyboardMarkup(keyboard)

    def get_user_management_menu(self):
        keyboard = [
            [InlineKeyboardButton("封禁用户", callback_data='action_ban_user')],
            [InlineKeyboardButton("解封用户", callback_data='action_unban_user')],
            [InlineKeyboardButton("添加控制面板管理员", callback_data='action_add_panel_admin')],
            [InlineKeyboardButton("移除控制面板管理员", callback_data='action_remove_panel_admin')],
            [InlineKeyboardButton("返回主菜单", callback_data='menu_main')]
        ]
        return InlineKeyboardMarkup(keyboard)

    def get_ai_models_menu(self):
        keyboard = [[InlineKeyboardButton(model_info["name"], callback_data=f'model_{model_key}')] 
                    for model_key, model_info in AI_MODELS.items()]
        keyboard.append([InlineKeyboardButton("返回主菜单", callback_data='menu_main')])
        return InlineKeyboardMarkup(keyboard)

    def get_system_settings_menu(self):
        current_level = logging.getLogger().level
        keyboard = [
            [InlineKeyboardButton(f"日志级别: {'DEBUG' if current_level == logging.DEBUG else 'INFO'}", 
                                  callback_data='toggle_log_level')],
            [InlineKeyboardButton("更新群组管理员列表", callback_data='action_update_admins')],
            [InlineKeyboardButton("返回主菜单", callback_data='menu_main')]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def handle_menu(self, query):
        menu = query.data.split('_')[1]
        if menu == 'main':
            await query.edit_message_text("请选择一个选项：", reply_markup=self.get_main_menu())
        elif menu == 'toggles':
            await query.edit_message_text("功能开关设置：", reply_markup=self.get_toggles_menu())
        elif menu == 'user_management':
            await query.edit_message_text("用户管理：", reply_markup=self.get_user_management_menu())
        elif menu == 'ai_models':
            await query.edit_message_text("选择 AI 模型：", reply_markup=self.get_ai_models_menu())
        elif menu == 'system_settings':
            await query.edit_message_text("系统设置：", reply_markup=self.get_system_settings_menu())
        elif menu == 'stats':
            stats = self.control_panel.bot.user_manager.get_stats()
            await query.edit_message_text(f"统计信息：\n{stats}", reply_markup=self.get_main_menu())

# 从 message_handlers.py
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received message from user {update.effective_user.id} in chat {update.effective_chat.id}")
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if update.effective_chat.type == 'private':
        await handle_private_message(update, context)
        return

    if chat_id not in context.bot_data['managed_chats'] or chat_id in context.bot_data['paused_chats']:
        logger.info(f"Chat {chat_id} is not managed or paused, ignoring message")
        return

    # 检查用户是否是管理员
    is_admin = await context.bot.user_manager.is_admin(chat_id, user_id)
    if is_admin and context.bot.features['ignore_admins']:
        logger.info(f"User {user_id} is admin and ignore_admins is enabled, ignoring message")
        return  # 如果是管理员且启用了忽略管理员功能，直接返回

    user = update.effective_user
    user_info = json.dumps({
        "id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "join_date": await context.bot.user_manager.get_join_date(chat_id, user.id),
        "message_count": await context.bot.user_manager.get_message_count(chat_id, user.id)
    })
    message_text = update.message.text

    if context.bot.features['spam_detection']:
        try:
            logger.info(f"Performing spam detection for message from user {user_id}")
            spam_result = await context.bot.spam_detector.is_spam(user_info, message_text)
            
            if spam_result['state'] == 1:
                logger.info(f"Spam detected from user {user_id}")
                await update.message.delete()
                if context.bot.features['auto_ban'] and await context.bot.spam_detector.should_ban(spam_result):
                    logger.info(f"Auto-banning user {user_id}")
                    await context.bot.user_manager.ban_user(chat_id, user.id)
                    await update.effective_chat.ban_member(user.id)
                    if not context.bot.features['silent_mode']:
                        await update.effective_chat.send_message(f"用户 {user.mention_html()} 已被封禁，因为发送了垃圾消息。\n原因：{spam_result['spam_reason']}", parse_mode='HTML')
                else:
                    logger.info(f"Warning user {user_id}")
                    warning = await context.bot.user_manager.warn_user(chat_id, user.id)
                    if not context.bot.features['silent_mode']:
                        await update.effective_chat.send_message(f"检测到来自用户 {user.mention_html()} 的可疑消息。\n原因：{spam_result['spam_reason']}\n这是第 {warning} 次警告。", parse_mode='HTML')
            else:
                logger.info(f"Message from user {user_id} is not spam")
        except Exception as e:
            logger.error(f"垃圾检测出错: {str(e)}")
            await notify_admin(context.bot, f"垃圾检测出错: {str(e)}")
            
    await context.bot.user_manager.increment_message_count(chat_id, user.id)

async def welcome_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"New member(s) joined chat {update.effective_chat.id}")
    chat_id = update.effective_chat.id
    if chat_id not in context.bot_data['managed_chats'] or chat_id in context.bot_data['paused_chats']:
        return
    if not context.bot.features['welcome_message'] or context.bot.features['silent_mode']:
        return
    for new_member in update.message.new_chat_members:
        await update.effective_chat.send_message(f"欢迎 {new_member.mention_html()} 加入群组！", parse_mode='HTML')
        await context.bot.user_manager.add_new_user(chat_id, new_member.id)

async def update_chat_admins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Updating admin list for chat {update.effective_chat.id}")
    chat_id = update.effective_chat.id
    if chat_id in context.bot_data['managed_chats']:
        try:
            admins = await context.bot.get_chat_administrators(chat_id)
            admin_ids = [admin.user.id for admin in admins]
            await context.bot.user_manager.update_admins(chat_id, admin_ids)
        except Exception as e:
            logger.error(f"更新管理员列表出错: {str(e)}")
            await notify_admin(context.bot, f"更新管理员列表出错: {str(e)}")

async def handle_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Received private message from user {update.effective_user.id}")
    user_id = update.effective_user.id
    if user_id not in PANEL_ADMINS:
        await update.message.reply_text("抱歉，只有管理员才能私聊使用机器人。")
        return

    if context.bot.control_panel:
        await context.bot.control_panel.process_user_input(update, context)
    else:
        await update.message.reply_text("控制面板尚未初始化，请稍后再试。")

    if not context.user_data.get('awaiting_input'):
        # 如果不在等待用户输入的状态，执行垃圾检测
        user_info = json.dumps({
            "id": user_id,
            "username": update.effective_user.username,
            "first_name": update.effective_user.first_name,
            "last_name": update.effective_user.last_name,
            "join_date": "N/A",  # 私聊没有加入日期
            "message_count": 0  # 私聊不计数
        })
        message_text = update.message.text
        try:
            logger.info(f"Performing spam detection for private message from user {user_id}")
            spam_result = await context.bot.spam_detector.is_spam(user_info, message_text)
            result_text = f"垃圾检测结果：\n状态: {'垃圾消息' if spam_result['state'] == 1 else '正常消息'}\n"
            result_text += f"垃圾概率: {spam_result['spam_score']}%\n"
            result_text += f"原因: {spam_result['spam_reason']}"
            await update.message.reply_text(result_text)
        except Exception as e:
            logger.error(f"私聊垃圾检测出错: {str(e)}")
            await update.message.reply_text("垃圾检测过程中出现错误，请稍后再试。")
            await notify_admin(context.bot, f"私聊垃圾检测出错: {str(e)}")
            
import json
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from ai_models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class KeywordDetector:
    def __init__(self):
        self.keywords = [
            '广告', '推广', '优惠', '折扣', '限时', '抢购', '秒杀', '免费',  # 中文关键词
            'ad', 'promo', 'discount', 'limited', 'offer', 'free', 'cheap', 'buy now', 'viagra'  # 英文关键词
        ]

    def detect(self, message_text):
        message_text = message_text.lower()  # 转换为小写以进行不区分大小写的匹配
        score = sum(1 for keyword in self.keywords if keyword in message_text)
        spam_score = min(score * 20, 100)  # 每个关键词增加20分，最高100分
        return {
            "state": 1 if spam_score >= 40 else 0,  # 降低阈值到40
            "spam_score": spam_score,
            "spam_reason": "本地关键词检测" if spam_score >= 40 else "未检测到垃圾内容"
        }

    def is_spam(self, message_text):
        result = self.detect(message_text)
        return result["state"] == 1

class AIDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def detect(self, user_info, message_text):
        prompt = f"""
    user_info: "{user_info}"
    text: |
      以下是判断发言是否为垃圾广告的条件
      1.对于新加入群组的用户(加入时间不到1天，发言次数少于3次)，如果他的发言较短，我们会谨慎判断，降低识别为垃圾广告的概率，以免错误封禁。
      2.对于新加入群组的用户，如果他的头几次发布的信息存在很明显的垃圾广告特征(如使用区块链、赌博、色情、金融等关键词，或使用谐音、错别字、同音字等变体来规避关键字检测，或在聊天内容中插入符号、特殊字符、emoji等来混淆信息)，我们应当正确识别并判断，以免错误封禁。
      3.对于群组中已存在的用户(加入群组时间超过1天，发言次数超过3次)，如果他的发言字数较短且没有明显垃圾广告特征，我们应强制认定其发言不是垃圾广告，以免错误封禁。
      4.如果用户的名称中也存在明显的垃圾广告特征，我们也应当提高判定为垃圾广告的概率。
    
      垃圾广告特征示例:
      - 包含虚假支付机构或银行卡信息，如冒牌支付机构、虚假银行卡购买等；
      - 诱导用户加入群组、点击链接或参与虚假活动;
      - 涉及非法支付、赌博、贩卖禁止物品等违法活动;
      - 提供非法服务，如代开飞机会员、代付、刷单、赌台、出U、贷款、色粉、网赚、交友等。
    
      请根据以上信息和垃圾广告特征，对用户发言进行判断。
      
      这是该用户的基本资料:{user_info}
      
      双引号内的内容是一条来自该用户的发言:"{message_text}"
      
      根据以上信息，这条发言是垃圾广告或推广信息吗?
      
      请以以下 JSON 结构返回分析结果:
      {{"state":{{填写0或1，1表示是垃圾广告，0表示不是}},"spam_score":{{填写一个0-100的数字，表示垃圾广告的概率}},"spam_reason":{{判断是否为垃圾广告，并提供原因}}}}
      请替换花括号中的内容，并以"纯文本"形式直接回答上述的JSON对象，不要包含任何其他的文本。
    """

        response = await self.model_manager.generate_text(prompt)
        result = json.loads(response)
        return result

class SpamDetector:
    def __init__(self, model_manager):
        self.ai_detector = AIDetector(model_manager)
        self.keyword_detector = KeywordDetector()

    @classmethod
    async def create(cls):
        model_manager = ModelManager()
        return cls(model_manager)

    async def is_spam(self, user_info, message_text):
        try:
            result = await self.ai_detector.detect(user_info, message_text)
        except Exception as e:
            logger.error(f"AI 垃圾检测失败: {str(e)}")
            result = self.keyword_detector.detect(message_text)
        return result

    def should_ban(self, spam_result):
        return spam_result['state'] == 1 and spam_result['spam_score'] > 80
    
# 导入所需的库
import logging
import os
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from spam_detector.spam_detector import SpamDetector
from user_management.user_manager import UserManager
from ai_models.model_manager import ModelManager
from config.feature_config import FEATURES
from handlers.command_handlers import start, help, show_control_panel, warn_user, set_api_key
from handlers.message_handlers import handle_message, welcome_new_member, update_chat_admins
from utils.error_handler import error_handler
from handlers.menu_handlers import MenuHandlers
from handlers.action_handlers import ActionHandlers
from handlers.input_handlers import InputHandlers

# 配置日志
logger = logging.getLogger(__name__)

# SpamSentinelBot 类
class SpamSentinelBot:
    def __init__(self, application: Application):
        self.application = application
        self.spam_detector = None
        self.user_manager = None
        self.model_manager = None
        self.features = FEATURES
        self.managed_chats = set()
        self.paused_chats = {}
        self.control_panel = None
        logger.info("哨兵机器人已初始化")

    async def initialize(self):
        logger.info("正在初始化哨兵机器人组件")
        self.spam_detector = await SpamDetector.create()
        self.user_manager = await UserManager.create()
        self.model_manager = ModelManager()
        self.control_panel = ControlPanel(self)
        await self.control_panel.setup_handlers()
        await self.setup_handlers()
        
        self.application.bot_data['paused_chats'] = self.paused_chats
        
        logger.info("哨兵机器人初始化完成")

    async def setup_handlers(self):
        logger.info("正在设置消息处理程序")
        self.application.add_handler(CommandHandler("start", start))
        self.application.add_handler(CommandHandler("help", help))
        self.application.add_handler(CommandHandler("control", show_control_panel))
        self.application.add_handler(CommandHandler("warn", warn_user))
        self.application.add_handler(CommandHandler("set_api_key", set_api_key))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_member))
        self.application.add_handler(MessageHandler(filters.ChatType.GROUPS, update_chat_admins))
        self.application.add_error_handler(error_handler)
        logger.info("消息处理程序设置完成")

    async def toggle_feature(self, feature):
        if feature in self.features:
            self.features[feature] = not self.features[feature]
            return self.features[feature]
        return None

# ControlPanel 类
class ControlPanel:
    def __init__(self, bot):
        self.bot = bot
        self.menu_handlers = MenuHandlers(self)
        self.action_handlers = ActionHandlers(self)
        self.input_handlers = InputHandlers(self)

    async def setup_handlers(self):
        self.bot.application.add_handler(CallbackQueryHandler(self.button_callback))

    async def show_control_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.bot.user_manager.is_panel_admin(update.effective_user.id):
            await update.message.reply_text("抱歉，只有控制面板管理员可以访问控制面板。")
            return

        await update.message.reply_text("欢迎使用 SpamSentinel 控制面板。请选择一个选项：", 
                                        reply_markup=self.menu_handlers.get_main_menu())

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if not self.bot.user_manager.is_panel_admin(query.from_user.id):
            await query.edit_message_text("抱歉，只有控制面板管理员可以使用控制面板。")
            return

        action = query.data.split('_')[0]
        if action == 'menu':
            await self.menu_handlers.handle_menu(query)
        elif action == 'toggle':
            await self.action_handlers.handle_toggle(query)
        elif action == 'model':
            await self.action_handlers.handle_model_selection(query, context)
        elif action == 'action':
            await self.action_handlers.handle_action(query, context)

    async def process_user_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.input_handlers.process_user_input(update, context)

    def toggle_feature(self, feature):
        FEATURES[feature] = not FEATURES[feature]
        return FEATURES[feature]

# 全局变量和初始化函数
application = None
bot = None

async def initialize_application():
    global application, bot
    if application is None:
        logger.info("正在初始化应用程序")
        token = os.environ.get("BOT_TOKEN")
        if token is None:
            logger.error("未设置 BOT_TOKEN 环境变量")
            raise ValueError("未设置 BOT_TOKEN 环境变量")
        application = Application.builder().token(token).build()
        bot = SpamSentinelBot(application)
        logger.info("应用程序初始化成功")

def get_application():
    global application
    return application

def get_bot():
    global bot
    return bot

# 路由
router = APIRouter()

@router.post('/webhook/{token}')
async def webhook(token: str, request: Request):
    logger.info(f"Received webhook request for token: {token[:5]}...")
    application = get_application()
    bot = get_bot()
    
    if application and application.bot.token == token:
        logger.info("Token verified, processing update")
        try:
            update_data = await request.json()
            logger.debug(f"Received update data: {update_data}")
            update = Update.de_json(update_data, application.bot)
            await bot.application.process_update(update)
            logger.info("Update processed successfully")
            return Response(content='OK', status_code=200)
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")
            return Response(content='Internal Server Error', status_code=500)
    else:
        logger.warning("Unauthorized webhook access attempt")
        return Response(content='Unauthorized', status_code=401)

@router.get('/', response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Received request for home page")
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SpamSentinel Bot</title>
    </head>
    <body>
        <h1>SpamSentinel Bot</h1>
        <p>Bot is running. Use Telegram to interact with it.</p>
    </body>
    </html>
    '''
    return HTMLResponse(content=html)
import json
import os
from datetime import datetime, timedelta
from config import PANEL_ADMINS

class UserDataOperations:
    def __init__(self):
        self.chat_data = {}
        self.load_data()

    async def ban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        self.chat_data[chat_id]['banned_users'].add(user_id)
        self.save_data()

    async def unban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        self.chat_data[chat_id]['banned_users'].discard(user_id)
        self.save_data()

    async def add_new_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        if user_id not in self.chat_data[chat_id]['users']:
            self.chat_data[chat_id]['users'][user_id] = {
                'join_date': datetime.now().isoformat(),
                'message_count': 0,
                'last_message_date': None,
                'warning_count': 0
            }
        self.save_data()

    async def increment_message_count(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id in self.chat_data and user_id in self.chat_data[chat_id]['users']:
            self.chat_data[chat_id]['users'][user_id]['message_count'] += 1
            self.chat_data[chat_id]['users'][user_id]['last_message_date'] = datetime.now().isoformat()
            self.save_data()

    def get_join_date(self, chat_id, user_id):
        chat_id = str(chat_id)
        return self.chat_data.get(chat_id, {}).get('users', {}).get(user_id, {}).get('join_date', datetime.now().isoformat())

    def get_message_count(self, chat_id, user_id):
        chat_id = str(chat_id)
        return self.chat_data.get(chat_id, {}).get('users', {}).get(user_id, {}).get('message_count', 0)

    async def is_new_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data or user_id not in self.chat_data[chat_id]['users']:
            return True
        join_date = datetime.fromisoformat(self.chat_data[chat_id]['users'][user_id]['join_date'])
        message_count = self.chat_data[chat_id]['users'][user_id]['message_count']
        return (datetime.now() - join_date < timedelta(days=1)) and (message_count < 3)

    async def warn_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        if user_id not in self.chat_data[chat_id]['users']:
            await self.add_new_user(chat_id, user_id)
        self.chat_data[chat_id]['users'][user_id]['warning_count'] += 1
        self.save_data()
        return self.chat_data[chat_id]['users'][user_id]['warning_count']

    def save_data(self):
        with open('user_data.json', 'w') as f:
            json.dump({"chat_data": self.chat_data}, f, default=str)

    def load_data(self):
        if os.path.exists('user_data.json'):
            with open('user_data.json', 'r') as f:
                data = json.load(f)
                self.chat_data = data.get("chat_data", {})
        else:
            self.chat_data = {}

class AdminOperations:
    def __init__(self, user_data_ops):
        self.user_data_ops = user_data_ops
        self.panel_admins = set(PANEL_ADMINS)
        self.load_data()

    def is_panel_admin(self, user_id):
        return user_id in self.panel_admins

    async def is_admin(self, chat_id, user_id):
        chat_id = str(chat_id)
        return user_id in self.user_data_ops.chat_data.get(chat_id, {}).get('admins', [])

    async def update_admins(self, chat_id, admin_ids):
        chat_id = str(chat_id)
        if chat_id not in self.user_data_ops.chat_data:
            self.user_data_ops.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        self.user_data_ops.chat_data[chat_id]['admins'] = admin_ids
        self.save_data()

    def add_panel_admin(self, user_id):
        self.panel_admins.add(user_id)
        self.save_data()

    def remove_panel_admin(self, user_id):
        self.panel_admins.discard(user_id)
        self.save_data()

    def save_data(self):
        with open('admin_data.json', 'w') as f:
            json.dump({"panel_admins": list(self.panel_admins)}, f)

    def load_data(self):
        if os.path.exists('admin_data.json'):
            with open('admin_data.json', 'r') as f:
                data = json.load(f)
                self.panel_admins = set(data.get("panel_admins", []))
        else:
            self.panel_admins = set(PANEL_ADMINS)

class StatsManager:
    def __init__(self, user_data_ops):
        self.user_data_ops = user_data_ops

    def get_stats(self):
        total_users = sum(len(chat['users']) for chat in self.user_data_ops.chat_data.values())
        total_banned = sum(len(chat['banned_users']) for chat in self.user_data_ops.chat_data.values())
        total_admins = sum(len(chat['admins']) for chat in self.user_data_ops.chat_data.values())
        total_messages = sum(
            sum(user['message_count'] for user in chat['users'].values())
            for chat in self.user_data_ops.chat_data.values()
        )
        total_warnings = sum(
            sum(user['warning_count'] for user in chat['users'].values())
            for chat in self.user_data_ops.chat_data.values()
        )
        active_users = sum(
            sum(1 for user in chat['users'].values() if user['last_message_date'] and
                datetime.now() - datetime.fromisoformat(user['last_message_date']) < timedelta(days=7))
            for chat in self.user_data_ops.chat_data.values()
        )
        return f"""总用户数：{total_users}
群组管理员数：{total_admins}
被封禁用户数：{total_banned}
总消息数：{total_messages}
总警告数：{total_warnings}
近7天活跃用户数：{active_users}"""

class UserManager:
    def __init__(self):
        self.user_data_ops = UserDataOperations()
        self.admin_ops = AdminOperations(self.user_data_ops)
        self.stats_manager = StatsManager(self.user_data_ops)

    @classmethod
    async def create(cls):
        manager = cls()
        manager.load_data()
        return manager

    def is_panel_admin(self, user_id):
        return self.admin_ops.is_panel_admin(user_id)

    async def is_admin(self, chat_id, user_id):
        return await self.admin_ops.is_admin(chat_id, user_id)

    async def update_admins(self, chat_id, admin_ids):
        await self.admin_ops.update_admins(chat_id, admin_ids)

    async def ban_user(self, chat_id, user_id):
        await self.user_data_ops.ban_user(chat_id, user_id)

    async def unban_user(self, chat_id, user_id):
        await self.user_data_ops.unban_user(chat_id, user_id)

    async def add_new_user(self, chat_id, user_id):
        await self.user_data_ops.add_new_user(chat_id, user_id)

    async def increment_message_count(self, chat_id, user_id):
        await self.user_data_ops.increment_message_count(chat_id, user_id)

    def get_join_date(self, chat_id, user_id):
        return self.user_data_ops.get_join_date(chat_id, user_id)

    def get_message_count(self, chat_id, user_id):
        return self.user_data_ops.get_message_count(chat_id, user_id)

    async def is_new_user(self, chat_id, user_id):
        return await self.user_data_ops.is_new_user(chat_id, user_id)

    async def warn_user(self, chat_id, user_id):
        return await self.user_data_ops.warn_user(chat_id, user_id)

    def get_stats(self):
        return self.stats_manager.get_stats()

    def add_panel_admin(self, user_id):
        self.admin_ops.add_panel_admin(user_id)

    def remove_panel_admin(self, user_id):
        self.admin_ops.remove_panel_admin(user_id)

    def save_data(self):
        self.user_data_ops.save_data()
        self.admin_ops.save_data()

    def load_data(self):
        self.user_data_ops.load_data()
        self.admin_ops.load_data()
import json
import os
from datetime import datetime, timedelta
from config import PANEL_ADMINS

class UserDataOperations:
    def __init__(self):
        self.chat_data = {}
        self.load_data()

    async def ban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        self.chat_data[chat_id]['banned_users'].add(user_id)
        self.save_data()

    async def unban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        self.chat_data[chat_id]['banned_users'].discard(user_id)
        self.save_data()

    async def add_new_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        if user_id not in self.chat_data[chat_id]['users']:
            self.chat_data[chat_id]['users'][user_id] = {
                'join_date': datetime.now().isoformat(),
                'message_count': 0,
                'last_message_date': None,
                'warning_count': 0
            }
        self.save_data()

    async def increment_message_count(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id in self.chat_data and user_id in self.chat_data[chat_id]['users']:
            self.chat_data[chat_id]['users'][user_id]['message_count'] += 1
            self.chat_data[chat_id]['users'][user_id]['last_message_date'] = datetime.now().isoformat()
            self.save_data()

    def get_join_date(self, chat_id, user_id):
        chat_id = str(chat_id)
        return self.chat_data.get(chat_id, {}).get('users', {}).get(user_id, {}).get('join_date', datetime.now().isoformat())

    def get_message_count(self, chat_id, user_id):
        chat_id = str(chat_id)
        return self.chat_data.get(chat_id, {}).get('users', {}).get(user_id, {}).get('message_count', 0)

    async def is_new_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data or user_id not in self.chat_data[chat_id]['users']:
            return True
        join_date = datetime.fromisoformat(self.chat_data[chat_id]['users'][user_id]['join_date'])
        message_count = self.chat_data[chat_id]['users'][user_id]['message_count']
        return (datetime.now() - join_date < timedelta(days=1)) and (message_count < 3)

    async def warn_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        if user_id not in self.chat_data[chat_id]['users']:
            await self.add_new_user(chat_id, user_id)
        self.chat_data[chat_id]['users'][user_id]['warning_count'] += 1
        self.save_data()
        return self.chat_data[chat_id]['users'][user_id]['warning_count']

    def save_data(self):
        with open('user_data.json', 'w') as f:
            json.dump({"chat_data": self.chat_data}, f, default=str)

    def load_data(self):
        if os.path.exists('user_data.json'):
            with open('user_data.json', 'r') as f:
                data = json.load(f)
                self.chat_data = data.get("chat_data", {})
        else:
            self.chat_data = {}

class AdminOperations:
    def __init__(self, user_data_ops):
        self.user_data_ops = user_data_ops
        self.panel_admins = set(PANEL_ADMINS)
        self.load_data()

    def is_panel_admin(self, user_id):
        return user_id in self.panel_admins

    async def is_admin(self, chat_id, user_id):
        chat_id = str(chat_id)
        return user_id in self.user_data_ops.chat_data.get(chat_id, {}).get('admins', [])

    async def update_admins(self, chat_id, admin_ids):
        chat_id = str(chat_id)
        if chat_id not in self.user_data_ops.chat_data:
            self.user_data_ops.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        self.user_data_ops.chat_data[chat_id]['admins'] = admin_ids
        self.save_data()

    def add_panel_admin(self, user_id):
        self.panel_admins.add(user_id)
        self.save_data()

    def remove_panel_admin(self, user_id):
        self.panel_admins.discard(user_id)
        self.save_data()

    def save_data(self):
        with open('admin_data.json', 'w') as f:
            json.dump({"panel_admins": list(self.panel_admins)}, f)

    def load_data(self):
        if os.path.exists('admin_data.json'):
            with open('admin_data.json', 'r') as f:
                data = json.load(f)
                self.panel_admins = set(data.get("panel_admins", []))
        else:
            self.panel_admins = set(PANEL_ADMINS)

class StatsManager:
    def __init__(self, user_data_ops):
        self.user_data_ops = user_data_ops

    def get_stats(self):
        total_users = sum(len(chat['users']) for chat in self.user_data_ops.chat_data.values())
        total_banned = sum(len(chat['banned_users']) for chat in self.user_data_ops.chat_data.values())
        total_admins = sum(len(chat['admins']) for chat in self.user_data_ops.chat_data.values())
        total_messages = sum(
            sum(user['message_count'] for user in chat['users'].values())
            for chat in self.user_data_ops.chat_data.values()
        )
        total_warnings = sum(
            sum(user['warning_count'] for user in chat['users'].values())
            for chat in self.user_data_ops.chat_data.values()
        )
        active_users = sum(
            sum(1 for user in chat['users'].values() if user['last_message_date'] and
                datetime.now() - datetime.fromisoformat(user['last_message_date']) < timedelta(days=7))
            for chat in self.user_data_ops.chat_data.values()
        )
        return f"""总用户数：{total_users}
群组管理员数：{total_admins}
被封禁用户数：{total_banned}
总消息数：{total_messages}
总警告数：{total_warnings}
近7天活跃用户数：{active_users}"""

class UserManager:
    def __init__(self):
        self.user_data_ops = UserDataOperations()
        self.admin_ops = AdminOperations(self.user_data_ops)
        self.stats_manager = StatsManager(self.user_data_ops)

    @classmethod
    async def create(cls):
        manager = cls()
        manager.load_data()
        return manager

    def is_panel_admin(self, user_id):
        return self.admin_ops.is_panel_admin(user_id)

    async def is_admin(self, chat_id, user_id):
        return await self.admin_ops.is_admin(chat_id, user_id)

    async def update_admins(self, chat_id, admin_ids):
        await self.admin_ops.update_admins(chat_id, admin_ids)

    async def ban_user(self, chat_id, user_id):
        await self.user_data_ops.ban_user(chat_id, user_id)

    async def unban_user(self, chat_id, user_id):
        await self.user_data_ops.unban_user(chat_id, user_id)

    async def add_new_user(self, chat_id, user_id):
        await self.user_data_ops.add_new_user(chat_id, user_id)

    async def increment_message_count(self, chat_id, user_id):
        await self.user_data_ops.increment_message_count(chat_id, user_id)

    def get_join_date(self, chat_id, user_id):
        return self.user_data_ops.get_join_date(chat_id, user_id)

    def get_message_count(self, chat_id, user_id):
        return self.user_data_ops.get_message_count(chat_id, user_id)

    async def is_new_user(self, chat_id, user_id):
        return await self.user_data_ops.is_new_user(chat_id, user_id)

    async def warn_user(self, chat_id, user_id):
        return await self.user_data_ops.warn_user(chat_id, user_id)

    def get_stats(self):
        return self.stats_manager.get_stats()

    def add_panel_admin(self, user_id):
        self.admin_ops.add_panel_admin(user_id)

    def remove_panel_admin(self, user_id):
        self.admin_ops.remove_panel_admin(user_id)

    def save_data(self):
        self.user_data_ops.save_data()
        self.admin_ops.save_data()

    def load_data(self):
        self.user_data_ops.load_data()
        self.admin_ops.load_data()