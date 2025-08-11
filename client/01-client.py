#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import argparse
import os
from PIL import Image
import time
from rich import print
from pathlib import Path
import traceback

def test_health(base_url):
    """测试健康检查接口"""
    url = f"{base_url}/health"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[red]健康检查失败: {e}[/red]")
        return None

def analyze_image(base_url, image_path, request_type="<DETAILED_CAPTION>", max_tokens=512, user_id="test_user"):
    """发送图片到 Florence API 进行分析"""
    url = f"{base_url}/analyze_image"
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"[red]图片不存在: {image_path}[/red]")
        return None
    
    # 准备文件和表单数据
    files = {
        'image_file': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
    }
    
    data = {
        'request_type': request_type,
        'max_tokens': max_tokens,
        'user_id': user_id
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        
        print(f"[green]请求成功! 耗时: {elapsed_time:.2f} 秒[/green]")
        print(f"[bold green]返回结果: {response.json()}[/bold green]")
        return response.json()
    except Exception as e:
        print(f"[red]分析图片失败: {e}[/red]")
        print(traceback.format_exc())
        return None
    finally:
        # 关闭文件
        files['image_file'][1].close()

def display_result(result):
    """美观地显示结果"""
    if not result:
        return
    
    print("\n[bold yellow]====== 图像分析结果 ======[/bold yellow]")
    print(f"请求类型: {result.get('request_type')}")
    print(f"用户ID: {result.get('user_id')}")
    print("\n[bold green]图像描述:[/bold green]")
    
    # 处理描述文本 - 一般返回的是列表，取第一个元素
    descriptions = result.get('image_description')
    # if isinstance(descriptions, list) and descriptions:
    #     description = descriptions[0]
    # else:
    #     description = descriptions
        
    print(f"[cyan]{descriptions}[/cyan]")
    print("\n[bold yellow]========================[/bold yellow]")

def main():
    parser = argparse.ArgumentParser(description="Florence 图像描述 API 测试客户端")
    parser.add_argument("--server", type=str, default="http://localhost:28001", help="API 服务器地址")
    parser.add_argument("--image_file", type=str, required=False, help="要分析的图片路径", default="./images/707b99243b.png")
    parser.add_argument("--request_type", type=str, default="<MORE_DETAILED_CAPTION>", 
                        choices=[
                            "<CAPTION>", 
                            "<DETAILED_CAPTION>", 
                            "<MORE_DETAILED_CAPTION>", 
                            "<OD>", 
                            "<DENSE_REGION_CAPTION>", 
                            "<REGION_PROPOSAL>", 
                            "<CAPTION_TO_PHRASE_GROUNDING>", 
                            "<REFERRING_EXPRESSION_SEGMENTATION>"
                        ],
                        help="请求类型")
    parser.add_argument("--max_tokens", type=int, default=1024, help="生成的最大标记数")
    parser.add_argument("--user_id", type=str, default="test_user", help="用户ID")
    
    args = parser.parse_args()
    
    # 测试健康检查
    print(f"[yellow]测试健康检查接口...[/yellow]")
    health_result = test_health(args.server)
    if health_result:
        print(f"[green]健康检查成功: {health_result}[/green]")
    else:
        print("[red]健康检查失败，退出测试[/red]")
        return
    
    # 测试图像分析
    print(f"[yellow]开始分析图片: {args.image_file}[/yellow]")
    result = analyze_image(
        base_url=args.server,
        image_path=args.image_file,
        request_type=args.request_type,
        max_tokens=args.max_tokens,
        user_id=args.user_id
    )
    
    # 显示结果
    display_result(result)

if __name__ == "__main__":
    main()
