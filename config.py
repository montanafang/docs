import os

# AI Model API Keys (初始化为空字符串，稍后通过API设置)
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

# 控制面板管理员ID列表
PANEL_ADMINS = [int(id) for id in os.getenv('PANEL_ADMINS', '').split(',') if id]

# 功能开关
FEATURES = {
    'spam_detection': True,
    'auto_ban': True,
    'welcome_message': True,
    'user_stats': True,
    'warning_system': True,
    'ignore_admins': True,
    'silent_mode': False,
}

# 垃圾检测配置
SPAM_DETECTION = {
    'min_chars': 5,  # 最小字符数
    'max_emojis': 5,  # 最大emoji数
    'max_urls': 2,  # 最大URL数
    'banned_words': ['广告', '推广', '优惠', '折扣', '限时', '抢购', '秒杀', '免费'],  # 禁用词列表
    'sensitivity': 0.7,  # 敏感度（0-1）
}

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