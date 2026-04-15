#!/usr/bin/env python3
"""
测试模型调用的故障转移机制

验证系统在 LongCat API 失败时能够自动切换到其他提供商。
"""

import os
import time
from src.infrastructure.provider_manager import get_provider_manager
from src.infrastructure.model_caller import make_model_caller


def test_model_caller():
    """测试模型调用器的故障转移机制"""
    print("=== 测试模型调用器故障转移机制 ===")
    print()
    
    # 获取供应商管理器
    pm = get_provider_manager()
    pm.list_providers()
    print()
    
    # 创建模型调用器
    model_caller = make_model_caller()
    print("✅ 模型调用器创建成功")
    print()
    
    # 测试消息
    test_messages = [
        {
            "role": "user",
            "content": "请简要介绍一下你自己，以及你能做什么。"
        }
    ]
    
    print("📡 发送测试请求...")
    start_time = time.time()
    
    try:
        response = model_caller(test_messages)
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  响应时间: {response_time:.2f} 秒")
        print(f"📦 响应长度: {len(response)} 字符")
        print()
        
        if response.strip():
            print("✅ 模型调用成功!")
            print()
            print("📄 响应内容:")
            print("-" * 50)
            print(response)
            print("-" * 50)
        else:
            print("❌ 响应内容为空")
            
    except Exception as e:
        print(f"❌ 模型调用失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=== 测试完成 ===")


def test_provider_failover():
    """测试供应商故障转移"""
    print("\n=== 测试供应商故障转移 ===")
    print()
    
    # 获取供应商管理器
    pm = get_provider_manager()
    
    # 测试每个供应商
    providers = ["longcat", "aliyun_bailian", "anthropic"]
    
    for provider_name in providers:
        print(f"测试供应商: {provider_name}")
        try:
            if pm.set_provider(provider_name):
                # 测试健康检查
                current_provider = pm.get_provider()
                if current_provider:
                    print(f"  ✅ 供应商设置成功: {current_provider.name}")
                else:
                    print(f"  ❌ 供应商设置失败")
            else:
                print(f"  ❌ 无法设置供应商: {provider_name}")
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
        print()


def main():
    """主函数"""
    # 测试模型调用器
    test_model_caller()
    
    # 测试供应商故障转移
    test_provider_failover()


if __name__ == "__main__":
    main()