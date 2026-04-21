#!/usr/bin/env python3
"""扫描SSH端口并查看容器内目录"""

import subprocess
import sys

# 根据之前的经验，端口通常在30000-50000范围
# 先尝试一些常见的端口
ports_to_try = list(range(35000, 35100)) + list(range(36000, 36100)) + list(range(37000, 37100))

print("开始扫描SSH端口...")
print(f"将尝试 {len(ports_to_try)} 个端口")
print()

found_port = None

for port in ports_to_try:
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=2",
             "-i", r"C:\Users\trae\.ssh\id_ed25519",
             "-p", str(port),
             "root@10.176.37.31",
             "ls -la /workspace/"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print(f"\n✅ 成功连接到端口 {port}!")
            print("=" * 60)
            print("目录内容:")
            print(result.stdout)
            print("=" * 60)
            found_port = port
            break
        else:
            # 端口连接失败，继续下一个
            if port % 10 == 0:
                print(f"  已尝试到端口 {port}...", end="\r")
    except Exception as e:
        if port % 10 == 0:
            print(f"  已尝试到端口 {port}...", end="\r")
        continue

if found_port:
    print(f"\n找到可用端口: {found_port}")

    # 继续查看日志文件
    print("\n查看日志文件...")
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-i", r"C:\Users\trae\.ssh\id_ed25519",
             "-p", str(found_port),
             "root@10.176.37.31",
             "cat /workspace/results.log 2>/dev/null | head -100 || echo 'No results.log'"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
    except Exception as e:
        print(f"读取日志失败: {e}")

    sys.exit(0)
else:
    print("\n❌ 未找到可用SSH端口")
    print("容器可能还在启动中，请稍后再试")
    sys.exit(1)
