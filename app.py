import asyncio
from flask import Flask, request, jsonify
from typing import Dict, List
import re
import random

app = Flask(__name__)

# 模拟数据存储
users_data: Dict[str, Dict[str, any]] = {}
chat_data: Dict[str, Dict[str, List[str]]] = {}
FEATURES = {
    "spam_detection": True,
    "user_management": True
}
AI_MODELS = {
    "gpt3": {"name": "GPT-3", "api_key": None},
    "gpt4": {"name": "GPT-4", "api_key": None}
}
current_model = "gpt3"

# 垃圾检测功能
async def is_spam(user_info: Dict, message_text: str) -> Dict:
    # 简单的垃圾检测逻辑
    spam_keywords = ["buy now", "click here", "limited offer"]
    is_spam = any(keyword in message_text.lower() for keyword in spam_keywords)
    confidence = 0.9 if is_spam else 0.1
    return {"is_spam": is_spam, "confidence": confidence}

# 用户管理功能
class UserManager:
    @staticmethod
    async def ban_user(chat_id: str, user_id: str) -> None:
        if chat_id not in chat_data:
            chat_data[chat_id] = {"banned_users": []}
        if user_id not in chat_data[chat_id]["banned_users"]:
            chat_data[chat_id]["banned_users"].append(user_id)

    @staticmethod
    async def unban_user(chat_id: str, user_id: str) -> None:
        if chat_id in chat_data and "banned_users" in chat_data[chat_id]:
            chat_data[chat_id]["banned_users"] = [u for u in chat_data[chat_id]["banned_users"] if u != user_id]

    @staticmethod
    async def warn_user(chat_id: str, user_id: str) -> int:
        if user_id not in users_data:
            users_data[user_id] = {"warnings": 0}
        users_data[user_id]["warnings"] += 1
        return users_data[user_id]["warnings"]

    @staticmethod
    def get_stats() -> Dict:
        total_users = len(users_data)
        total_chats = len(chat_data)
        total_banned = sum(len(chat.get("banned_users", [])) for chat in chat_data.values())
        return {
            "total_users": total_users,
            "total_chats": total_chats,
            "total_banned": total_banned
        }

user_manager = UserManager()

# API 端点
@app.route('/api/spam_check', methods=['POST'])
async def spam_check():
    data = request.json
    user_info = data.get('user_info')
    message_text = data.get('message_text')
    
    if not user_info or not message_text:
        return jsonify({"error": "Missing user_info or message_text"}), 400
    
    if FEATURES["spam_detection"]:
        result = await is_spam(user_info, message_text)
        return jsonify(result)
    else:
        return jsonify({"error": "Spam detection feature is disabled"}), 403

@app.route('/api/ban_user', methods=['POST'])
async def ban_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    if FEATURES["user_management"]:
        await user_manager.ban_user(chat_id, user_id)
        return jsonify({"message": "User banned successfully"})
    else:
        return jsonify({"error": "User management feature is disabled"}), 403

@app.route('/api/unban_user', methods=['POST'])
async def unban_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    if FEATURES["user_management"]:
        await user_manager.unban_user(chat_id, user_id)
        return jsonify({"message": "User unbanned successfully"})
    else:
        return jsonify({"error": "User management feature is disabled"}), 403

@app.route('/api/warn_user', methods=['POST'])
async def warn_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    if FEATURES["user_management"]:
        warning_count = await user_manager.warn_user(chat_id, user_id)
        return jsonify({"message": f"User warned. Warning count: {warning_count}"})
    else:
        return jsonify({"error": "User management feature is disabled"}), 403

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    if FEATURES["user_management"]:
        stats = user_manager.get_stats()
        return jsonify({"stats": stats})
    else:
        return jsonify({"error": "User management feature is disabled"}), 403

@app.route('/api/toggle_feature', methods=['POST'])
def toggle_feature():
    data = request.json
    feature = data.get('feature')
    
    if not feature or feature not in FEATURES:
        return jsonify({"error": "Invalid feature"}), 400
    
    FEATURES[feature] = not FEATURES[feature]
    return jsonify({"feature": feature, "state": FEATURES[feature]})

@app.route('/api/set_model', methods=['POST'])
def set_model():
    global current_model
    data = request.json
    model_key = data.get('model_key')
    api_key = data.get('api_key')
    
    if not model_key or not api_key:
        return jsonify({"error": "Missing model_key or api_key"}), 400
    
    if model_key not in AI_MODELS:
        return jsonify({"error": "Invalid model_key"}), 400
    
    AI_MODELS[model_key]["api_key"] = api_key
    current_model = model_key
    return jsonify({"message": f"{AI_MODELS[model_key]['name']} model set successfully"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
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