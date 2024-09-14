from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/spam_check', methods=['POST'])
async def spam_check():
    data = request.json
    user_info = data.get('user_info')
    message_text = data.get('message_text')
    
    if not user_info or not message_text:
        return jsonify({"error": "Missing user_info or message_text"}), 400
    
    result = await spam_detector.is_spam(user_info, message_text)
    return jsonify(result)

@app.route('/api/ban_user', methods=['POST'])
async def ban_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    await user_manager.ban_user(chat_id, user_id)
    return jsonify({"message": "User banned successfully"})

@app.route('/api/unban_user', methods=['POST'])
async def unban_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    await user_manager.unban_user(chat_id, user_id)
    return jsonify({"message": "User unbanned successfully"})

@app.route('/api/warn_user', methods=['POST'])
async def warn_user():
    data = request.json
    chat_id = data.get('chat_id')
    user_id = data.get('user_id')
    
    if not chat_id or not user_id:
        return jsonify({"error": "Missing chat_id or user_id"}), 400
    
    warning_count = await user_manager.warn_user(chat_id, user_id)
    return jsonify({"message": f"User warned. Warning count: {warning_count}"})

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    stats = user_manager.get_stats()
    return jsonify({"stats": stats})

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)