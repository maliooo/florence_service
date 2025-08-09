from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException  # uv pip install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import io
from PIL import Image  # uv pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
import torch  # uv pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
from transformers import AutoProcessor, AutoModelForCausalLM  # uv pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
from rich import print  # uv pip install rich -i https://pypi.tuna.tsinghua.edu.cn/simple
import time
import argparse
import uvicorn  # uv pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
from pathlib import Path
import asyncio
import base64

# 参考：https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
PROMPT_TASK_LIST = [
    "<CAPTION>", 
    "<DETAILED_CAPTION>", 
    "<MORE_DETAILED_CAPTION>", 
    "<OD>", 
    "<DENSE_REGION_CAPTION>", 
    "<REGION_PROPOSAL>", 
    "<CAPTION_TO_PHRASE_GROUNDING>", 
    "<REFERRING_EXPRESSION_SEGMENTATION>"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=Path(__file__).parents[1] / "models/Florence-2-large")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--weight_type", type=str, default="fp16", choices=["fp16", "auto"])
    parser.add_argument("--prompt_task", type=str, default="<MORE_DETAILED_CAPTION>", choices=PROMPT_TASK_LIST)
    parser.add_argument("--max_tokens", type=int, default=1024)
    return parser.parse_args()

args = parse_args()
print(f"[bold green]启动参数为：{args}[/bold green]")


# 定义请求数据模型
class ImageRequestData(BaseModel):
    description: Optional[str] = None
    user_id: Optional[str] = None
    request_type: Optional[str] = args.prompt_task
    max_tokens: Optional[int] = args.max_tokens 

# 添加用于POST请求的数据模型
class ImageBase64RequestData(BaseModel):
    image: str  # base64编码的图像
    description: Optional[str] = None
    user_id: Optional[str] = None
    request_type: Optional[str] = args.prompt_task
    max_tokens: Optional[int] = args.max_tokens

app = FastAPI(title="Florence 图像描述 API")

# 添加 CORS 中间件, 作用是允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if args.weight_type == "fp16":
    model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_dir, 
                trust_remote_code=True,
                local_files_only=True
            ).to(torch.float16).to(args.device).eval()  # 加载模型, fp16
    print(f"[bold green]加载模型成功，模型类型为：{type(model)}, model_device为：{model.device}, model_dtype为：{model.dtype}[/bold green]")
else:
    model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_dir, 
                trust_remote_code=True,
                local_files_only=True
            ).to(args.device).eval()  # 加载模型, fp16
    print(f"[bold green]加载模型成功，模型类型为：{type(model)}, model_device为：{model.device}, model_dtype为：{model.dtype}[/bold green]")
processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)


# 创建信号量控制并发请求数
# 根据您的硬件资源调整最大并发数
MAX_CONCURRENT_REQUESTS = 1  # 根据GPU内存和性能调整
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# 依赖函数，用于解析表单数据
def get_image_request_data(
    description: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    request_type: Optional[str] = Form(args.prompt_task),
    max_tokens: Optional[int] = Form(args.max_tokens)
) -> ImageRequestData:
    return ImageRequestData(
        description=description,
        user_id=user_id,
        request_type=request_type,
        max_tokens=max_tokens
    )

@app.post("/analyze_image", response_model=dict)
async def analyze_image(
    image_file: UploadFile = File(...),
    request_data: ImageRequestData = Depends(get_image_request_data)
):
    # 尝试获取信号量，如果已达到最大并发数，将等待
    async with semaphore:
        try:
            t1 = time.time()
            prompt = None
            image_content = await image_file.read()
            image = Image.open(io.BytesIO(image_content))
            # pil_image.save("test.jpg")
            print(f"获得请求参数: {request_data}")
            
            # 准备提示词
            assert request_data.request_type in PROMPT_TASK_LIST

            if request_data.description is None:
                prompt = request_data.request_type
            else:
                prompt = f"{request_data.request_type}{request_data.description}"
            
            # 处理图像并生成描述
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            if args.weight_type == "fp16":
                inputs = inputs.to(torch.float16).to(args.device)
            else:
                inputs = inputs.to(args.device)
            
            # 生成描述
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=request_data.max_tokens,
                    do_sample=False,
                    num_beams=3,
                )
            
            # 解码输出
            # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=request_data.request_type, 
                image_size=(image.width, image.height)
            )
            print(f"[bold green]解析后的结果为：{parsed_answer}[/bold green]")

            # 返回结果
            return {
                "success": True,
                "image_description": generated_text,
                "user_id": request_data.user_id,
                "request_type": request_data.request_type,
                "prompt": prompt,
                "error": None,
                "time_cost": f"{(time.time() - t1):.2f}s",
                "parsed_answer": parsed_answer,
                "bboxes": None,
                "labels": None
            }
        except Exception as e:
            print(f"[red]获得请求参数: {request_data}\n发生错误: {e}[/red]")
            return {
                "success": False,
                "error": str(e),
                "image_description": None,
                "user_id": request_data.user_id,
                "request_type": request_data.request_type,
                "prompt": prompt,
                "time_cost": f"{(time.time() - t1):.2f}s"
            }

@app.post("/analyze_image_base64", response_model=dict)
async def analyze_image_base64(request_data: ImageBase64RequestData):
    # 尝试获取信号量，如果已达到最大并发数，将等待
    async with semaphore:
        try:
            t1 = time.time()
            prompt = None
            
            # 解码base64图像
            try:
                image_data = base64.b64decode(request_data.image)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"无效的base64图像: {str(e)}")
            
            print(f"获得请求参数: {request_data.request_type}, {request_data.description}, {request_data.user_id}, {request_data.max_tokens}")
            
            # 准备提示词
            assert request_data.request_type in PROMPT_TASK_LIST

            if request_data.description is None:
                prompt = request_data.request_type
            else:
                prompt = f"{request_data.request_type}{request_data.description}"
            
            # 处理图像并生成描述
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            if args.weight_type == "fp16":
                inputs = inputs.to(torch.float16).to(args.device)
            else:
                inputs = inputs.to(args.device)
            
            # 生成描述
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=request_data.max_tokens,
                    do_sample=False,
                    num_beams=3,
                )
            
            # 解码输出
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=request_data.request_type, 
                image_size=(image.width, image.height)
            )
            print(f"[bold green]解析后的结果为：{parsed_answer}[/bold green]")

            # 返回结果
            return {
                "success": True,
                "image_description": generated_text,
                "user_id": request_data.user_id,
                "request_type": request_data.request_type,
                "prompt": prompt,
                "error": None,
                "time_cost": f"{(time.time() - t1):.2f}s",
                "parsed_answer": parsed_answer,
                "bboxes": None,
                "labels": None
            }
        except Exception as e:
            print(f"[red]获得请求参数: {request_data}\n发生错误: {e}[/red]")
            return {
                "success": False,
                "error": str(e),
                "image_description": None,
                "user_id": request_data.user_id,
                "request_type": request_data.request_type,
                "prompt": prompt,
                "time_cost": f"{(time.time() - t1):.2f}s"
            }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"启动服务: host: {args.host}, port: {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
