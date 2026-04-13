"""测试供应商管理器"""

import os
import json
import pytest
from unittest.mock import patch
from pathlib import Path

from src.infrastructure.provider_manager import (
    ProviderManager, ProviderConfig, get_provider_manager
)


class TestProviderConfig:
    """测试供应商配置类"""

    def test_provider_config_creation(self):
        """测试供应商配置创建"""
        config_data = {
            "_provider": "test_provider",
            "_name": "Test Provider",
            "_description": "Test Description",
            "_models": {"default": "test-model"},
            "_endpoints": {"base": "https://test.com", "chat": "https://test.com/chat"},
            "_auth": {"env_var": "TEST_API_KEY"},
            "_headers": {"Authorization": "Bearer {api_key}"},
            "_request_format": {},
            "_supported_features": [],
            "_rate_limits": {},
            "_cost": {}
        }

        config = ProviderConfig(config_data)
        assert config.provider == "test_provider"
        assert config.name == "Test Provider"
        assert config.get_model("default") == "test-model"

    def test_get_api_key_from_env(self):
        """测试从环境变量获取API密钥"""
        config_data = {
            "_provider": "test",
            "_auth": {"env_var": "TEST_API_KEY"}
        }
        config = ProviderConfig(config_data)

        # 模拟环境变量
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"}):
            api_key = config.get_api_key()
            assert api_key == "test-key-123"

    def test_get_headers_with_api_key(self):
        """测试构建请求头"""
        config_data = {
            "_provider": "test",
            "_headers": {"Authorization": "Bearer {api_key}"}
        }
        config = ProviderConfig(config_data)

        headers = config.get_headers("test-key-123")
        assert headers["Authorization"] == "Bearer test-key-123"


class TestProviderManager:
    """测试供应商管理器"""

    @pytest.fixture
    def mock_provider_configs(self, tmp_path):
        """创建模拟的供应商配置文件"""
        config_dir = tmp_path / "providers"
        config_dir.mkdir()

        # 创建测试供应商配置
        test_config = {
            "_provider": "test_provider",
            "_name": "Test Provider",
            "_description": "Test Description",
            "_models": {"default": "test-model"},
            "_endpoints": {"base": "https://test.com", "chat": "https://test.com/chat"},
            "_auth": {"type": "api_key", "env_var": "TEST_API_KEY"},
            "_headers": {"Authorization": "Bearer {api_key}"},
            "_request_format": {},
            "_supported_features": [],
            "_rate_limits": {},
            "_cost": {}
        }

        config_file = config_dir / "test_provider.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)

        return config_dir

    def test_load_providers(self, mock_provider_configs):
        """测试加载供应商配置"""
        pm = ProviderManager(str(mock_provider_configs))
        assert "test_provider" in pm.providers
        assert pm.providers["test_provider"].name == "Test Provider"

    def test_detect_provider_with_env_var(self, mock_provider_configs):
        """测试检测配置了环境变量的供应商"""
        pm = ProviderManager(str(mock_provider_configs))

        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}):
            provider = pm.detect_provider()
            assert provider is not None
            assert provider.provider == "test_provider"

    def test_detect_provider_no_env_var(self, mock_provider_configs):
        """测试没有配置环境变量的情况"""
        pm = ProviderManager(str(mock_provider_configs))

        # 不设置环境变量
        provider = pm.detect_provider()
        assert provider is None

    def test_set_provider(self, mock_provider_configs):
        """测试手动设置供应商"""
        pm = ProviderManager(str(mock_provider_configs))

        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}):
            success = pm.set_provider("test_provider")
            assert success is True
            assert pm.current_provider.provider == "test_provider"

    def test_create_unified_config(self, mock_provider_configs):
        """测试创建统一配置"""
        pm = ProviderManager(str(mock_provider_configs))

        with patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"}):
            pm.set_provider("test_provider")
            config = pm.create_unified_config()

            assert config is not None
            assert config["provider"] == "test_provider"
            assert config["env"]["ANTHROPIC_AUTH_TOKEN"] == "test-key-123"
            assert config["env"]["ANTHROPIC_BASE_URL"] == "https://test.com/chat"
            assert config["env"]["ANTHROPIC_MODEL"] == "test-model"

    def test_list_providers(self, mock_provider_configs, capsys):
        """测试列出供应商"""
        pm = ProviderManager(str(mock_provider_configs))
        pm.list_providers()

        captured = capsys.readouterr()
        assert "Test Provider" in captured.out


class TestProviderManagerIntegration:
    """集成测试"""

    def test_real_provider_loading(self):
        """测试加载真实的供应商配置文件"""
        pm = ProviderManager()

        # 检查是否加载了标准供应商
        expected_providers = ["aliyun_bailian", "anthropic", "longcat"]
        loaded_providers = list(pm.providers.keys())

        for provider in expected_providers:
            if provider in loaded_providers:
                print(f"✅ 加载了 {provider} 配置")
            else:
                print(f"⚠️  未加载 {provider} 配置")

    def test_provider_priority_detection(self):
        """测试供应商优先级检测"""
        pm = ProviderManager()

        # 模拟多个环境变量
        env_vars = {
            "ANTHROPIC_API_KEY": "anthropic-key",
            "DASHSCOPE_API_KEY": "dashscope-key",
            "LONGCAT_API_KEY": "longcat-key"
        }

        with patch.dict(os.environ, env_vars):
            provider = pm.detect_provider()
            # 应该优先选择百炼
            assert provider is not None
            assert provider.provider == "aliyun_bailian"


def test_get_provider_manager_singleton():
    """测试单例模式"""
    pm1 = get_provider_manager()
    pm2 = get_provider_manager()
    assert pm1 is pm2


if __name__ == "__main__":
    # 运行简单测试
    print("🧪 运行供应商管理器测试...")

    # 测试真实配置加载
    pm = ProviderManager()
    print(f"✅ 加载了 {len(pm.providers)} 个供应商配置")

    for name, provider in pm.providers.items():
        print(f"  - {provider.name} ({name})")

    print("\n测试完成！")