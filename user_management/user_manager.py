import json
import os
from datetime import datetime, timedelta
from config import PANEL_ADMINS

class UserDataOperations:
    def __init__(self):
        self.chat_data = {}
        self.load_data()

    def ban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
        self.chat_data[chat_id]['banned_users'].add(user_id)
        self.save_data()

    def unban_user(self, chat_id, user_id):
        chat_id = str(chat_id)
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
    def create(cls):
        manager = cls()
        manager.load_data()
        return manager

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