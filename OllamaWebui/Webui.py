import ollama
from flask import Flask, render_template, request, Response, session, redirect, url_for
import uuid
import os
import json
import time
import re

CONVERSATION_FOLDER = 'conversations'

app = Flask(__name__)
app.secret_key = 'your_secret_key'

CONVERSATION_DIR = 'conversations'
os.makedirs(CONVERSATION_DIR, exist_ok=True)

VERSION = "1.1.6"

@app.context_processor
def inject_version():
    return dict(version=VERSION)

def get_conversation_id():
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    return session['conversation_id']

def format_code_blocks(content):
    content = content.replace('###', '<br>')
    code_block_pattern = r'```([a-zA-Z]*)\n([\s\S]*?)```'
    content = re.sub(code_block_pattern, lambda match: f'''
        <div class="code-block-container">
            <div class="code-header">
                <span class="code-lang">{match.group(1).strip()}</span>
                <button class="copy-button" onclick="copyCode({repr(match.group(2))})">复制代码</button>
            </div>
            <pre><code>{match.group(2).strip()}</code></pre>
        </div>
    ''', content)
    return content

def get_conversation_by_id(conversation_id):
    try:
        file_path = os.path.join(CONVERSATION_FOLDER, f'{conversation_id}.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        return conversation
    except (FileNotFoundError, json.JSONDecodeError):
        return None

@app.route('/create_new_conversation', methods=['GET'])
def create_new_conversation():
    session['conversation_id'] = str(uuid.uuid4())
    return redirect(url_for('index'))

def save_conversation(conversation_id, user_input, bot_response):
    conversation_file = os.path.join(CONVERSATION_DIR, f'{conversation_id}.json')
    if os.path.exists(conversation_file):
        with open(conversation_file, 'r', encoding='utf-8') as f:
            conversation_history = json.load(f)
    else:
        conversation_history = []
    conversation_history.append({'user': user_input, 'bot': bot_response})
    with open(conversation_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)

def get_conversation_context(conversation_id):
    conversation_file = os.path.join(CONVERSATION_DIR, f'{conversation_id}.json')
    if os.path.exists(conversation_file):
        with open(conversation_file, 'r', encoding='utf-8') as f:
            conversation_history = json.load(f)
        context = ""
        for entry in conversation_history:
            context += f"用户: {entry['user']}\n机器人: {entry['bot']}\n"
        return context
    return ""

def remove_think_tags(content):
    thinkStartIndex = content.find('<think>')
    thinkEndIndex = content.find('</think>')
    while thinkStartIndex != -1 and thinkEndIndex != -1:
        content = content[:thinkStartIndex] + content[thinkEndIndex + len('</think>'):]
        thinkStartIndex = content.find('<think>')
        thinkEndIndex = content.find('</think>')
    return content

def get_conversation_history(conversation_id):
    conversation_file = os.path.join(CONVERSATION_DIR, f'{conversation_id}.json')
    if os.path.exists(conversation_file):
        with open(conversation_file, 'r', encoding='utf-8') as f:
            conversation_history = json.load(f)
        formatted_history = []
        for entry in conversation_history:
            user_message = entry['user']
            bot_message = remove_think_tags(entry['bot'])
            formatted_history.append({'user': user_message, 'bot': bot_message})
        return formatted_history
    return []

def api_generate(text: str):
    conversation_id = get_conversation_id()
    conversation_context = get_conversation_context(conversation_id)
    full_prompt = conversation_context + f"用户: {text}\n机器人:"

    def generate():
        print(f'提问：{text}')
        retries = 0
        max_retries = 5
        content = ""

        while retries < max_retries:
            try:
                stream = ollama.generate(
                    stream=True,
                    model='deepseek-r1:14b',
                    prompt=full_prompt,
                )
                content = ""
                for chunk in stream:
                    if chunk['done']:
                        save_conversation(conversation_id, text, content)
                        print(f'生成回答：{content}')
                        yield f"{content}"
                        return
                    content += chunk['response']
                    if '<think>' in content:
                        content = remove_think_tags(content)
                if not content.strip():
                    retries += 1
                    print(f'没有收到有效回答，正在进行第 {retries} 次重试...')
                    time.sleep(1)
                else:
                    save_conversation(conversation_id, text, content)
                    print(f'生成回答：{content}')
                    yield f"{content}"
                    return
            except ollama._types.ResponseError as e:
                print(f"Error: {e}")
                yield "抱歉，服务不可用。请确保 Ollama 服务已启动并运行。"
                return
        print(f'未能成功生成有效回答，重试次数超过 {max_retries} 次')
        yield "抱歉，机器人未能生成有效的回答。"

    return Response(generate(), content_type='text/plain;charset=utf-8')

@app.route('/history')
def history():
    conversations = []
    for filename in os.listdir(CONVERSATION_DIR):
        if filename.endswith('.json'):
            conversation_id = filename.split('.')[0]
            conversations.append(conversation_id)
    return render_template('history.html', conversations=conversations)

@app.route('/conversation/<conversation_id>')
def view_conversation(conversation_id):
    conversation = get_conversation_by_id(conversation_id)
    if not conversation:
        return "Conversation not found", 404
    for msg in conversation:
        if msg.get('bot'):
            msg['bot'] = format_code_blocks(msg['bot'])
    return render_template('conversation.html', history=conversation, conversation_id=conversation_id)

@app.route('/')
def index():
    conversation_id = request.args.get('conversation_id', None)
    if conversation_id:
        session['conversation_id'] = conversation_id
        session['secret_key'] = str(uuid.uuid4())
    history = []
    if 'conversation_id' in session:
        history = get_conversation_history(session["conversation_id"])
    return render_template('index.html', history=history)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    return api_generate(question)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)