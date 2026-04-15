"""Provider Manager - 多供应商API配置管理器

支持多种LLM供应商的统一接口：
- 阿里云百炼 (DASHSCOPE)
- Anthropic Claude
- LongCat API
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ProviderConfig:
    """供应商配置类"""

    def __init__(self, config_data: Dict[str, Any]):
        self.provider = config_data["_provider"]
        self.name = config_data["_name"]
        self.description = config_data["_description"]
        self.models = config_data["_models"]
        self.endpoints = config_data["_endpoints"]
        self.auth = config_data["_auth"]
        self.headers = config_data["_headers"]
        self.request_format = config_data["_request_format"]
        self.supported_features = config_data["_supported_features"]
        self.rate_limits = config_data["_rate_limits"]
        self.cost = config_data.get("_cost", {})

    def get_api_key(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        env_var = self.auth["env_var"]
        return os.getenv(env_var)

    def get_model(self, model_type: str = "default") -> str:
        """获取指定类型的模型"""
        return self.models.get(model_type, self.models["default"])

    def get_headers(self, api_key: str) -> Dict[str, str]:
        """构建请求头"""
        headers = {}
        for key, value in self.headers.items():
            headers[key] = value.format(api_key=api_key)
        return headers

    def validate(self) -> bool:
        """验证配置完整性"""
        required_fields = ["_provider", "_models", "_endpoints", "_auth"]
        return all(hasattr(self, field.lstrip("_")) for field in required_fields)


class ProviderManager:
    """供应商管理器"""

    def __init__(self, config_dir: str = "config/providers"):
        self.config_dir = Path(config_dir)
        self.providers: Dict[str, ProviderConfig] = {}
        self.current_provider: Optional[ProviderConfig] = None
        self._load_providers()

    def _load_providers(self):
        """加载所有供应商配置"""
        if not self.config_dir.exists():
            print(f"⚠️  配置目录不存在: {self.config_dir}")
            return

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                provider = ProviderConfig(config_data)
                if provider.validate():
                    self.providers[provider.provider] = provider
                    print(f"OK 加载供应商配置: {provider.name}")
                else:
                    print(f"NO 配置验证失败: {config_file}")

            except Exception as e:
                print(f"NO 加载配置失败 {config_file}: {e}")

    def detect_provider(self) -> Optional[ProviderConfig]:
        """自动检测可用的供应商"""
        # 按优先级检测环境变量 - longcat优先于aliyun_bailian
        provider_priority = ["longcat", "aliyun_bailian", "anthropic"]

        for provider_name in provider_priority:
            provider = self.providers.get(provider_name)
            if provider and provider.get_api_key():
                print(f"AUTO 自动检测到供应商: {provider.name}")
                self.current_provider = provider
                return provider

        print("NO 未检测到任何配置的供应商")
        return None

    def set_provider(self, provider_name: str) -> bool:
        """手动设置供应商"""
        provider = self.providers.get(provider_name)
        if not provider:
            print(f"NO 未知供应商: {provider_name}")
            return False

        if not provider.get_api_key():
            print(f"NO 供应商 {provider.name} API密钥未配置")
            return False

        self.current_provider = provider
        print(f"OK 已切换到供应商: {provider.name}")
        return True

    def get_provider(self) -> Optional[ProviderConfig]:
        """获取当前供应商"""
        if not self.current_provider:
            return self.detect_provider()
        return self.current_provider

    def list_providers(self):
        """列出所有可用供应商"""
        print("📋 可用供应商:")
        for name, provider in self.providers.items():
            api_key = provider.get_api_key()
            status = "✅ 已配置" if api_key else "❌ 未配置"
            print(f"  - {provider.name} ({name}): {status}")

    def create_unified_config(self) -> Optional[Dict[str, Any]]:
        """创建统一配置格式"""
        provider = self.get_provider()
        if not provider:
            return None

        api_key = provider.get_api_key()
        if not api_key:
            print(f"❌ 供应商 {provider.name} API密钥未配置")
            return None

        return {
            "provider": provider.provider,
            "provider_name": provider.name,
            "env": {
                "ANTHROPIC_BASE_URL": provider.endpoints["chat"],
                "ANTHROPIC_AUTH_TOKEN": api_key,
                "ANTHROPIC_MODEL": provider.get_model("default"),
                "ANTHROPIC_REASONING_MODEL": provider.get_model("reasoning"),
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": provider.get_model("haiku"),
                "ANTHROPIC_DEFAULT_SONNET_MODEL": provider.get_model("sonnet"),
                "ANTHROPIC_DEFAULT_OPUS_MODEL": provider.get_model("opus")
            },
            "headers": provider.get_headers(api_key),
            "rate_limits": provider.rate_limits,
            "cost": provider.cost,
            "includeCoAuthoredBy": False,
            "effortLevel": "high"
        }


def get_provider_manager():
    """获取供应商管理器单例"""
    if not hasattr(get_provider_manager, "_instance"):
        get_provider_manager._instance = ProviderManager()
    return get_provider_manager._instance


if __name__ == "__main__":
    # 测试供应商管理器
    pm = ProviderManager()
    pm.list_providers()

    # 尝试自动检测
    provider = pm.detect_provider()
    if provider:
        config = pm.create_unified_config()
        print(f"\n统一配置已生成:")
        print(f"  供应商: {config['provider_name']}")
        print(f"  模型: {config['env']['ANTHROPIC_MODEL']}")
        print(f"  端点: {config['env']['ANTHROPIC_BASE_URL']}")
    else:
        print("\n❌ 无法检测到可用供应商")