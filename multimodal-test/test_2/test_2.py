import os
from openai import OpenAI
import base64

_api_key = os.getenv("KIMI_API_KEY")
if not _api_key:
    raise ValueError(
        "请设置环境变量 KIMI_API_KEY。例如在终端执行：\n"
        "  export KIMI_API_KEY=你的密钥"
    )
client = OpenAI(api_key=_api_key, base_url="https://api.moonshot.cn/v1")

def encode_image(image_path):
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, prompt="请详细描述这张图片的内容"):
    # 获取图片的 base64 编码
    base64_image = encode_image(image_path)
    
    # 构建消息，包含图片和文本
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # 调用 Kimi 多模态模型
    response = client.chat.completions.create(
        model="kimi-k2.5",  # 或指定具体版本如 "kimi-k2.5"
        messages=messages,
    )
    
    return response.choices[0].message.content


def build_content_for_api(content_list):
    """将 content 列表转换为 API 所需的 content 格式（文本 + 图片 base64）"""
    api_content = []
    for item in content_list:
        if item["type"] == "text":
            api_content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image":
            base64_image = encode_image(item["img_path"])
            api_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            if item.get("caption"):
                api_content.append({"type": "text", "text": f"[图注] {item['caption']}"})
    return api_content


def analyze_multimodal(content_list, prompt="请根据给出的文本和图片进行分析"):
    """根据 content 列表（文本+图片）调用多模态大模型"""
    api_content = build_content_for_api(content_list)
    if prompt:
        api_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": api_content}]
    print("messages:")
    print(messages)
    response = client.chat.completions.create(
        model="kimi-k2.5",
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    content = [
        {
            "type": "text",
            "text": "The effectiveness of Al in density reduction is almost the same in both the ferritic and austenitic alloys, since the coefficients for Al in Eqs. (1) and (2) are nearly identical (0.098 vs 0.101). This indicates a $1 . 3 \\%$ reduction in density per $1 \\%$ Al addition. The addition of C is very effective in density reduction for austenitic low density steels. The effectiveness of C is about four times higher than that of Al."
        },
        {
            "type": "image",
            "img_path": "/Users/mac/Desktop/paper_extractor/SteelDig/datasets/paper_parsered/1-s2.0-S007964251730066X-main/1-s2.0-S007964251730066X-main/auto/images/0f13110e1a8ceae7d07e3cfe9110ceb708d907d2200bcf6432704464de69d92c.jpg",
            "caption": "Fig. 1. Effect of alloy elements on the physical properties of ferritic iron; (a) density reduction of ferritic iron by elements lighter than Fe [24,27]; (b) the reduction of Young’s modulus of Fe-Al steels in the annealed state as a function of Al content [24,28]."
        }
    ]
    response = analyze_multimodal(
        content,
        prompt="请结合上述文本与图片，总结 Al、C 对密度的影响，并简要说明图中信息。"
    )
    print(response)
