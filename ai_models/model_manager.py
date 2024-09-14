import json
import requests
from abc import ABC, abstractmethod
from config import AI_MODELS

class AIModel(ABC):
    def __init__(self, api_key, model_name=None):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def generate_text(self, prompt):
        pass

class OpenAIModel(AIModel):
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.api_base = "https://api.openai.com/v1"

    def generate_text(self, prompt):
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

    def generate_text(self, prompt):
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

    def generate_text(self, prompt):
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

    def generate_text(self, prompt):
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

    def set_model(self, model_key: str, api_key: str) -> bool:
        if model_key not in AI_MODELS:
            return False

        model_config = AI_MODELS[model_key]
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

    def generate_text(self, prompt: str) -> str:
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model].generate_text(prompt)
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
                    self.set_model(self.current_model, AI_MODELS[self.current_model]["api_key"])
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