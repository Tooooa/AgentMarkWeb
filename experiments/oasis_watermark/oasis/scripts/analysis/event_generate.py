import json
from openai import OpenAI
import os

# Initialize client based on project context if possible, 
# but keep the original logic for compatibility
def _get_client():
    # Attempt to load from default config path
    try:
        # Check standard locations
        config_paths = [
            'config.json',
            'configs/social_media/config.json',
            '../../configs/social_media/config.json'
        ]
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return OpenAI(api_key=config['api_key'], base_url=config['base_url']), config
    except:
        pass
    return None, None

def format_history_prompt(history_events):
    """格式化历史视频事件"""
    if not history_events:
        return ""
    
    history_text = "\n已经生成过的视频内容："
    for i, event in enumerate(history_events, 1):
        history_text += f"\n{i}. {event}"
    history_text += "\n请生成一个与上述视频内容不同类型、不同主题的新视频。"
    return history_text

def generate_video_browse_prompt(name, profile, history_events=None):
    """生成视频浏览的prompt"""
    history_prompt = format_history_prompt(history_events) if history_events else ""
    
    return f"""
作为{name}，{profile}
{history_prompt}
请根据你的兴趣和个性特征，生成一个你可能会看到的视频内容。

1. 视频类型范围：
   - 生活日常
   - 美食探店
   - 旅游风景
   - 知识科普
   - 宠物萌宠
   - 科技数码
   - 游戏实况
   - 音乐舞蹈

2. 视频信息需要包含：
   - 视频标题
   - 视频时长（1-30分钟）
   - 视频作者信息
   - 视频简短描述
   - 当前播放量、点赞数、评论数
   - 视频标签（2-4个）

3. 输出格式（请严格按照此JSON格式输出）：
{{
    "title": "视频标题",
    "duration": "视频时长",
    "author": "作者名称",
    "description": "视频描述",
    "views": "播放量",
    "likes": "点赞数",
    "comments": "评论数",
    "tags": ["标签1", "标签2"]
}}

注意：
1. 生成的视频内容应该符合{name}的兴趣和性格特点
2. 只需要输出JSON格式的视频信息，不要包含其他内容
3. 确保数值合理，播放量、点赞数和评论数要符合正常比例
4. 生成的内容必须与历史记录中的视频有明显的差异
"""

def format_video_event_to_text(video_event_json):
    """将视频事件JSON转换为自然语言描述"""
    try:
        # 将字符串转换为JSON对象
        if isinstance(video_event_json, str):
            video_data = json.loads(video_event_json)
        else:
            video_data = video_event_json
            
        # 构建描述文本
        event_description = f"在推荐页面出现了一个视频《{video_data['title']}》。这是由{video_data['author']}制作的一个{video_data['duration']}的视频，主要{video_data['description']}视频目前已经获得了{video_data['views']}的播放量，{video_data['likes']}的点赞数，以及{video_data['comments']}条评论。这个视频被标记为{' 和 '.join(video_data['tags'])}。"
        
        return event_description
    except Exception as e:
        return f"解析视频事件时出错: {str(e)}"

def generate_video_event(client=None, config=None, temperature=0.9, presence_penalty=2, history_events=None, history_responses=None):
    """生成一个随机的视频浏览事件
    Args:
        client: OpenAI 客户端 (可选)
        config: 配置对象 (可选)
        temperature: 控制随机性的参数，范围0-2，越大越随机，默认0.9
        presence_penalty: 重复惩罚参数，范围0-2，越大越避免重复，默认2
        history_events: 历史视频事件列表
        history_responses: 历史响应列表 (Oasis run_experiment.py 会传递此参数)
    """
    if client is None or config is None:
        local_client, local_config = _get_client()
        client = client or local_client
        config = config or local_config
        
    if client is None:
        raise ValueError("OpenAI client and config must be provided or available in standard locations.")

    prompt = generate_video_browse_prompt(
        name=config['role_config']['name'],
        profile=config['role_config'].get('profile', config['role_config'].get('description', '')),
        history_events=history_events
    )
    
    messages = [
        {"role": "system", "content": config['role_config']['system_prompt'].format(
            name=config['role_config']['name'],
            profile=config['role_config'].get('profile', config['role_config'].get('description', ''))
        )},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=config.get('model', 'deepseek-chat'),
        messages=messages,
        temperature=temperature,
        presence_penalty=presence_penalty
    )
    
    # Check if the content is wrapped in code blocks
    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    return json.loads(content.strip())

if __name__ == "__main__":
    # 用于存储历史视频事件
    history_events = []
    
    # 生成次随机视频浏览事件
    try:
        for i in range(2):
            print(f"\n=== 视频浏览事件 {i+1} ===")
            video_event = generate_video_event(history_events=history_events)
            event_desc = format_video_event_to_text(video_event)
            
            # 添加到历史记录
            history_events.append(event_desc)
            
            print("原始数据:")
            print(json.dumps(video_event, indent=2, ensure_ascii=False))
            print("\n自然语言描述:")
            print(event_desc)
    except Exception as e:
        print(f"Error: {e}")
