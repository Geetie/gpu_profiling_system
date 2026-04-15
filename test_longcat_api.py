#!/usr/bin/env python3
"""
LongCat API 可调用性测试脚本

测试 LongCat API 的连接性和响应能力，验证大模型是否能够正常回复。
添加详细的网络诊断和调试信息。
"""

import os
import json
import time
import requests
import socket
import urllib3
from typing import Dict, List, Optional


def check_network_connectivity():
    """检查网络连接性"""
    print("=== 网络连接诊断 ===")
    
    # 1. 检查 DNS 解析
    print("1. DNS 解析测试:")
    try:
        ip = socket.gethostbyname("api.longcat.chat")
        print(f"   ✅ 成功解析 api.longcat.chat: {ip}")
    except socket.gaierror as e:
        print(f"   ❌ DNS 解析失败: {e}")
        return False
    
    # 2. 检查端口连通性
    print("2. 端口连通性测试:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex(("api.longcat.chat", 443))
        if result == 0:
            print("   ✅ 端口 443 连通")
        else:
            print(f"   ❌ 端口 443 连接失败: {result}")
            return False
        sock.close()
    except Exception as e:
        print(f"   ❌ 端口测试失败: {e}")
        return False
    
    # 3. 检查 HTTP 响应
    print("3. HTTP 响应测试:")
    try:
        response = requests.get("https://api.longcat.chat", timeout=15)
        print(f"   ✅ HTTP 响应状态码: {response.status_code}")
    except Exception as e:
        print(f"   ❌ HTTP 响应测试失败: {e}")
        return False
    
    print("✅ 网络连接正常")
    print()
    return True


def test_longcat_api():
    """测试 LongCat API 的可调用性"""
    print("=== LongCat API 可调用性测试 ===")
    print()
    
    # 1. 检查环境变量
    api_key = os.getenv("LONGCAT_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 LONGCAT_API_KEY 环境变量")
        print("请在环境变量中设置 LONGCAT_API_KEY")
        return False
    
    print("✅ 环境变量 LONGCAT_API_KEY 已设置")
    print(f"API Key 长度: {len(api_key)} 字符")
    print(f"API Key 前 8 位: {api_key[:8]}...")
    print()
    
    # 2. 构建请求
    base_url = "https://api.longcat.chat/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 简化测试消息，减少响应时间
    messages = [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
    
    payload = {
        "model": "LongCat-Flash-Thinking-2601",
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
    }
    
    print("📡 发送测试请求到 LongCat API...")
    print(f"端点: {base_url}")
    print(f"模型: {payload['model']}")
    print(f"请求大小: {len(json.dumps(payload))} 字节")
    print()
    
    # 3. 发送请求并测量时间
    start_time = time.time()
    
    # 禁用证书验证（临时测试）
    session = requests.Session()
    session.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        response = session.post(
            base_url,
            headers=headers,
            json=payload,
            timeout=15  # 减少超时时间
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  响应时间: {response_time:.2f} 秒")
        print(f"📊 状态码: {response.status_code}")
        print(f"📦 响应大小: {len(response.content)} 字节")
        print()
        
        if response.status_code == 200:
            print("✅ API 调用成功!")
            print()
            
            # 4. 解析响应
            try:
                result = response.json()
                print("📄 响应内容:")
                print("-" * 50)
                
                # 提取回复内容
                choices = result.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    role = message.get("role", "")
                    
                    print(f"角色: {role}")
                    print(f"内容: {content}")
                    print("-" * 50)
                    
                    # 检查 token 使用情况
                    usage = result.get("usage", {})
                    if usage:
                        print("📊 Token 使用情况:")
                        print(f"  - 输入: {usage.get('prompt_tokens', 0)}")
                        print(f"  - 输出: {usage.get('completion_tokens', 0)}")
                        print(f"  - 总计: {usage.get('total_tokens', 0)}")
                    
                    return True
                else:
                    print("❌ 错误: 响应中没有 choices 字段")
                    print(f"完整响应: {json.dumps(result, indent=2)}")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ 错误: 无法解析 JSON 响应: {e}")
                print(f"响应内容: {response.text[:500]}...")
                return False
        else:
            print(f"❌ API 调用失败: {response.status_code}")
            print(f"错误信息: {response.text[:500]}...")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        print("可能的原因:")
        print("1. 网络连接问题")
        print("2. LongCat API 服务器不可用")
        print("3. API Key 无效")
        print("4. 防火墙限制")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ 请求超时: {e}")
        print("API 响应时间超过 15 秒")
        print("可能的原因:")
        print("1. 网络延迟过高")
        print("2. LongCat API 服务器负载过高")
        print("3. 代理服务器问题")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        session.close()


def test_fallback_providers():
    """测试其他 API 提供商作为备选"""
    print("\n=== 备选 API 提供商测试 ===")
    print()
    
    # 测试阿里云百炼
    aliyun_key = os.getenv("DASHSCOPE_API_KEY")
    if aliyun_key:
        print("✅ 环境变量 DASHSCOPE_API_KEY 已设置")
        print(f"API Key 长度: {len(aliyun_key)} 字符")
        print(f"API Key 前 8 位: {aliyun_key[:8]}...")
    else:
        print("⚠️  未设置 DASHSCOPE_API_KEY 环境变量")
    
    # 测试 Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("✅ 环境变量 ANTHROPIC_API_KEY 已设置")
        print(f"API Key 长度: {len(anthropic_key)} 字符")
        print(f"API Key 前 8 位: {anthropic_key[:8]}...")
    else:
        print("⚠️  未设置 ANTHROPIC_API_KEY 环境变量")
    
    print()


def main():
    """主函数"""
    # 测试网络连接
    network_ok = check_network_connectivity()
    
    # 测试 LongCat API
    longcat_success = False
    if network_ok:
        longcat_success = test_longcat_api()
    else:
        print("⚠️  网络连接测试失败，跳过 LongCat API 测试")
    
    # 测试备选提供商
    test_fallback_providers()
    
    print("=== 测试完成 ===")
    if longcat_success:
        print("✅ LongCat API 可正常调用!")
    else:
        print("❌ LongCat API 调用失败，请检查配置和网络连接")
        print("建议:")
        print("1. 检查网络连接是否正常")
        print("2. 验证 LONGCAT_API_KEY 是否正确")
        print("3. 尝试使用阿里云 API 作为备选")
        print("4. 检查防火墙是否允许访问 api.longcat.chat")
    

if __name__ == "__main__":
    main()