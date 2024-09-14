import json
import logging
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

class SpamDetector:
    def __init__(self, model_manager):
        self.ai_detector = AIDetector(model_manager)
        self.keyword_detector = KeywordDetector()

    @classmethod
    def create(cls):
        model_manager = ModelManager()
        return cls(model_manager)

    def is_spam(self, user_info, message_text):
        try:
            result = self.ai_detector.detect(user_info, message_text)
        except Exception as e:
            logger.error(f"AI 垃圾检测失败: {str(e)}")
            result = self.keyword_detector.detect(message_text)
        return result

    def should_ban(self, spam_result):
        return spam_result['state'] == 1 and spam_result['spam_score'] > 80