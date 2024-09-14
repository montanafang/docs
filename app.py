import json
import logging
import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from tenacity import retry, stop_after_attempt, wait_fixed

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
FEATURES = {
    'spam_detection': True,
    'auto_ban': True,
    'welcome_message': True,
    'user_stats': True,
    'warning_system': True,
    'ignore_admins': True,
    'silent_mode': False,
    'user_management': True
}

AI_MODELS = {
    "openai": {"name": "OpenAI", "api_key": ""},
    "openrouter": {"name": "OpenRouter", "api_key": ""},
    "cloudflare": {"name": "Cloudflare AI", "api_key": ""},
    "google": {"name": "Google Gemini", "api_key": ""},
    "custom_openai": {"name": "Custom OpenAI Endpoint", "api_key": "", "endpoint": ""}
}

PANEL_ADMINS = []

# 用户数据操作
class UserDataOperations:
    def __init__(self):
        self.chat_data = {}
        self.load_data()

    def ban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        self.chat_data[chat_id]['banned_users'].add(user_id)
        self.save_data()

    def unban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id in self.chat_data:
            self.chat_data[chat_id]['banned_users'].discard(user_id)
            self.save_data()

    def add_new_user(self, chat_id, user_id):
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

    def increment_message_count(self, chat_id, user_id):
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

    def is_new_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data or user_id not in self.chat_data[chat_id]['users']:
            return True
        join_date = datetime.fromisoformat(self.chat_data[chat_id]['users'][user_id]['join_date'])
        message_count = self.chat_data[chat_id]['users'][user_id]['message_count']
        return (datetime.now() - join_date < timedelta(days=1)) and (message_count < 3)

    def warn_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        if chat_id not in self.chat_data:
            self.chat_data[chat_id] = {'users': {}, 'admins': [], 'banned_users': set(), 'warnings': {}}
        if user_id not in self.chat_data[chat_id]['users']:
            self.add_new_user(chat_id, user_id)
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

# 管理员操作
class AdminOperations:
    def __init__(self, user_data_ops):
        self.user_data_ops = user_data_ops
        self.panel_admins = set(PANEL_ADMINS)
        self.load_data()

    def is_panel_admin(self, user_id):
        return user_id in self.panel_admins

    def is_admin(self, chat_id, user_id):
        chat_id = str(chat_id)
        return user_id in self.user_data_ops.chat_data.get(chat_id, {}).get('admins', [])

    def update_admins(self, chat_id, admin_ids):
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

# 统计管理
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
            sum(user.get('warning_count', 0) for user in chat['users'].values())
            for chat in self.user_data_ops.chat_data.values()
        )
        active_users = sum(
            sum(1 for user in chat['users'].values() if user.get('last_message_date') and
                datetime.now() - datetime.fromisoformat(user['last_message_date']) < timedelta(days=7))
            for chat in self.user_data_ops.chat_data.values()
        )
        return {
            "total_users": total_users,
            "total_admins": total_admins,
            "total_banned": total_banned,
            "total_messages": total_messages,
            "total_warnings": total_warnings,
            "active_users_7d": active_users
        }

# 用户管理
class UserManager:
    def __init__(self):
        self.user_data_ops = UserDataOperations()
        self.admin_ops = AdminOperations(self.user_data_ops)
        self.stats_manager = StatsManager(self.user_data_ops)

    def is_panel_admin(self, user_id):
        return self.admin_ops.is_panel_admin(user_id)

    def is_admin(self, chat_id, user_id):
        return self.admin_ops.is_admin(chat_id, user_id)

    def update_admins(self, chat_id, admin_ids):
        self.admin_ops.update_admins(chat_id, admin_ids)

    def ban_user(self, chat_id, user_id):
        self.user_data_ops.ban_user(chat_id, user_id)

    def unban_user(self, chat_id, user_id):
        self.user_data_ops.unban_user(chat_id, user_id)

    def add_new_user(self, chat_id, user_id):
        self.user_data_ops.add_new_user(chat_id, user_id)

    def increment_message_count(self, chat_id, user_id):
        self.user_data_ops.increment_message_count(chat_id, user_id)

    def get_join_date(self, chat_id, user_id):
        return self.user_data_ops.get_join_date(chat_id, user_id)

    def get_message_count(self, chat_id, user_id):
        return self.user_data_ops.get_message_count(chat_id, user_id)

    def is_new_user(self, chat_id, user_id):
        return self.user_data_ops.is_new_user(chat_id, user_id)

    def warn_user(self, chat_id, user_id):
        return self.user_data_ops.warn_user(chat_id, user_id)

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

# 关键词检测器
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

# AI 检测器
class AIDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def detect(self, user_info, message_text):
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

        response = self.model_manager.generate_text(prompt)
        result = json.loads(response)
        return result

# 垃圾检测器
class SpamDetector:
    def __init__(self, model_manager):
        self.ai_detector = AIDetector(model_manager)
        self.keyword_detector = KeywordDetector()

    def is_spam(self, user_info, message_text):
        try:
            result = self.ai_detector.detect(user_info, message_text)
        except Exception as e:
            logger.error(f"AI 垃圾检测失败: {str(e)}")
            result = self.keyword_detector.detect(message_text)
        return result

    def should_ban(self, spam_result):
        return spam_result['state'] == 1 and spam_result['spam_score'] > 80

# 模型管理器
class ModelManager:
    def __init__(self):
        self.current_model = None

    def set_model(self, model_key, api_key):
        if model_key in AI_MODELS:
            AI_MODELS[model_key]["api_key"] = api_key
            self.current_model = model_key
            return True
        return False

    def generate_text(self, prompt):
        if not self.current_model or self.current_model not in AI_MODELS:
            raise ValueError("No model selected or model not found")
        
        # 这里应该实现实际的 API 调用逻辑
        # 由于我们没有实际的 API 密钥，这里只是返回一个模拟的响应
        return json.dumps({
            "state": 0,
            "spam_score": 10,
            "spam_reason": "This is a simulated response from the AI model."
        })

# 初始化
user_manager = UserManager()
model_manager = ModelManager()
spam_detector = SpamDetector(model_manager)

# API 路由
@app.route('/api/spam_check', methods=['POST'])
def spam_check():
    data = request.json
    user_info = data.get('user_info')
    message_text = data.get('message_text')
    
    if not user_info or not message_text:
        return jsonify({"error": "Missing user_info or message_text"}), 400
    
    if FEATURES["spam_detection"]:
        result = spam_detector.is_spam(user_info, message_text)
        return jsonify(result)
    else:
        return jsonify({"error": "Spam detection feature is disabled"}), 403

@app.route('/api/ban_user', methods=['POST'])
def ban_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    if FEATURES["user_management"]:
        user_manager.ban_user(chat_id, user_id)
        return jsonify({"message": "User banned successfully"})
    else:
        return jsonify({"error": "User management feature is disabled"}), 403

@app.route('/api/unban_user', methods=['POST'])
def unban_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    if FEATURES["user_management"]:
        user_manager.unban_user(chat_id, user_id)
        return jsonify({"message": "User unbanned successfully"})
    else:
        return jsonify({"error": "User management feature is disabled"}), 403

@app.route('/api/warn_user', methods=['POST'])
def warn_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    if FEATURES["warning_system"]:
        warning_count = user_manager.warn_user(chat_id, user_id)
        return jsonify({"message": f"User warned. Warning count: {warning_count}"})
    else:
        return jsonify({"error": "Warning system feature is disabled"}), 403

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    if FEATURES["user_stats"]:
        stats = user_manager.get_stats()
        return jsonify({"stats": stats})
    else:
        return jsonify({"error": "User stats feature is disabled"}), 403

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
    data = request.json
    model_key = data.get('model_key')
    api_key = data.get('api_key')
    
    if not model_key or not api_key:
        return jsonify({"error": "Missing model_key or api_key"}), 400
    
    if model_key not in AI_MODELS:
        return jsonify({"error": "Invalid model_key"}), 400
    
    success = model_manager.set_model(model_key, api_key)
    if success:
        return jsonify({"message": f"{AI_MODELS[model_key]['name']} model set successfully"})
    else:
        return jsonify({"error": "Failed to set model"}), 500

@app.route('/api/add_panel_admin', methods=['POST'])
def add_panel_admin():
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    
    user_manager.add_panel_admin(user_id)
    return jsonify({"message": f"User {user_id} added as panel admin successfully"})

@app.route('/api/remove_panel_admin', methods=['POST'])
def remove_panel_admin():
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    
    user_manager.remove_panel_admin(user_id)
    return jsonify({"message": f"User {user_id} removed from panel admins successfully"})

@app.route('/api/update_admins', methods=['POST'])
def update_admins():
    data = request.json
    chat_id = data.get('chat_id')
    admin_ids = data.get('admin_ids')
    
    if not chat_id or not admin_ids:
        return jsonify({"error": "Missing chat_id or admin_ids"}), 400
    
    user_manager.update_admins(chat_id, admin_ids)
    return jsonify({"message": "Admin list updated successfully"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)