import time
import re

# 装饰函数，打印函数运行时间
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"消耗时间: {func.__name__}, {(time.time()-start):.2f} seconds.")
        return res
    return wrapper


# florence 字符串后处理
def str_postprocess(text):
    if text is None:
        return ""
    if str(text).strip() == "":
        return ""

    text = str(text).strip()
    if str(text).lower().startswith("The image shows a 3D model of".lower()):
        text = text[len("The image shows a 3D model of"):]
    elif str(text).lower().startswith("The image shows a detailed sketch of".lower()):
        text = text[len("The image shows a detailed sketch of"):]
    elif str(text).lower().startswith("The image shows a rendering of".lower()):
        text = text[len("The image shows a rendering of"):]
    elif str(text).lower().startswith("The image shows a 3D model of".lower()):
        text = text[len("The image shows a 3D model of"):]
    elif str(text).lower().startswith("The image shows".lower()):
        text = text[len("The image shows"):]
    elif str(text).lower().startswith("a computer screen with a drawing of".lower()):
        text = text[len("a computer screen with a drawing of"):]


    text = text.strip()

    if str(text).lower().startswith("a 3D rendering of".lower()):
        text = text[len("a 3D rendering of"):]
    elif str(text).lower().startswith("a drawing of".lower()):
        text = text[len("a drawing of"):]
    elif str(text).lower().startswith("a sketch of".lower()):
        text = text[len("a sketch of"):]

    # caption会出现的一些特殊情况
    if str(text).lower().startswith("A rendering of ".lower()):
        text = text[len("A rendering of "):]

    # more detail caption
    if str(text).lower().startswith("The image is a black and white line drawing of".lower()):
        text = text[len("The image is a black and white line drawing of"):]
    if str(text).lower().startswith("The image is a black and white sketch of".lower()):
        text = text[len("The image is a black and white sketch of"):]
    # The image is a 3D rendering of a modern living room.
    if str(text).lower().startswith("The image is a 3D rendering of".lower()):
        text = text[len("The image is a 3D rendering of"):]
        
    # --------------------------- 替换
    if "a 3D rendering of".lower() in str(text).lower():
        text = text.replace("a 3D rendering of", "")
    if "a black and white line drawing of".lower() in str(text).lower():
        text = text.replace("a black and white line drawing of", "")
    if "a sketch of".lower() in str(text).lower():
        text = text.replace("a sketch of", "")
    if "a drawing of".lower() in str(text).lower():
        text = text.replace("a drawing of", "")
    
    text = text.strip(" .")  # 去除首尾空格和句号
    return text

def str_color_postprocess(text):
    """去除字符串中的颜色信息"""

    
    # 定义一个颜色列表
    colors = ['black', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'gray', 'beige', 'teal']

    if text is None:
        return ""
    if str(text).strip() == "":
        return ""

    text = str(text).strip()
    # 定义一个函数，用于替换包含颜色单词的任何变体
    def remove_color_variations(match):
        return ''

    # 使用正则表达式替换包含颜色单词的任何变体
    cleaned_text = text
    for color in colors:
        # 构建正则表达式，匹配任何包含颜色单词的变体
        pattern = r'\b' + re.escape(color) + r'\w*\b'
        cleaned_text = re.sub(pattern, remove_color_variations, cleaned_text, flags=re.IGNORECASE)
        # 打印结果
    cleaned_text = cleaned_text.replace('  ', ' ')
    return cleaned_text

