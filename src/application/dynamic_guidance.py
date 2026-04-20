"""
DynamicGuidanceManager - 动态分层引导管理器

解决LLM在后期轮次产生"疲劳"的问题，通过动态调整引导消息的
强度和内容，确保指令遵循率保持在高水平。

设计原则:
- P3 上下文工程优于提示工程: 引导消息需动态组装
- P4 可组合性: 可独立使用或嵌入AgentLoop

分层策略:
1. INFO_LEVEL (token_weight=40): 正常引导 ("建议你...")
2. MANDATORY_LEVEL (token_weight=80): 强制引导 ("你必须...")
3. EMERGENCY_LEVEL (token_weight=150): 紧急恢复 ("🚨🚨🚨...")
"""

from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GuidanceLevel(Enum):
    """引导强度等级"""
    INFO = "info"
    MANDATORY = "mandatory"
    EMERGENCY = "emergency"


class DynamicGuidanceManager:
    """
    动态分层引导管理器
    
    根据当前轮次、停滞次数、目标进度等因素动态选择引导策略。
    """
    
    def __init__(self):
        self._turn_count: int = 0
        self._stall_count: int = 0
        self._target_progress: tuple[int, int] = (0, 0)
        
    def update_context(self, turn_count: int, stall_count: int,
                       progress: tuple[int, int]) -> None:
        """
        更新上下文信息
        
        Args:
            turn_count: 当前轮次
            stall_count: 连续无工具调用次数
            progress: (已完成数, 总数)
        """
        self._turn_count = turn_count
        self._stall_count = stall_count
        self._target_progress = progress
        
    def determine_level(self) -> GuidanceLevel:
        """
        根据当前状态确定引导级别
        
        策略规则:
        - Turn 1-5 + 无停滞 → INFO_LEVEL
        - Turn 6+ 或 有停滞 → MANDATORY_LEVEL  
        - 停滞>=2 或 Turn>15 → EMERGENCY_LEVEL
        - 所有目标完成 → COMPLETION_LEVEL
        """
        completed, total = self._target_progress
        
        # 所有目标完成
        if total > 0 and completed >= total:
            return GuidanceLevel.EMERGENCY  # 用于触发完成检测
            
        # 高停滞风险
        if self._stall_count >= 2 or self._turn_count > 15:
            return GuidanceLevel.EMERGENCY
            
        # 中等风险
        if self._stall_count >= 1 or self._turn_count > 10:
            return GuidanceLevel.MANDATORY
            
        # 正常情况
        return GuidanceLevel.INFO
    
    def get_token_weight(self, level: Optional[GuidanceLevel] = None) -> int:
        """获取指定级别的token权重"""
        if level is None:
            level = self.determine_level()
            
        weights = {
            GuidanceLevel.INFO: 40,
            GuidanceLevel.MANDATORY: 80,
            GuidanceLevel.EMERGENCY: 150
        }
        return weights.get(level, 40)
    
    def build_target_switch_guidance(self, target_name: str,
                                      design_principle: str,
                                      level: Optional[GuidanceLevel] = None) -> dict:
        """
        构建目标切换引导消息
        
        Args:
            target_name: 目标名称
            design_principle: 设计原则描述
            level: 指定级别(可选，默认自动确定)
            
        Returns:
            包含message和token_weight的字典
        """
        if level is None:
            level = self.determine_level()
            
        weight = self.get_token_weight(level)
        
        if level == GuidanceLevel.INFO:
            message = (
                f"Next target to measure: **{target_name}**\n\n"
                f"Design principle:\n{design_principle[:400]}\n\n"
                f"Please compile CUDA code for this target using compile_cuda."
            )
        elif level == GuidanceLevel.MANDATORY:
            message = (
                f"⚠️ MANDATORY: You MUST measure '{target_name}' next!\n\n"
                f"Remaining targets: {self._target_progress[1] - self._target_progress[0]}\n\n"
                f"Design principle:\n{design_principle[:500]}\n\n"
                f"Call compile_cuda NOW with new source code for {target_name}."
                f"Do NOT output text explanations."
            )
        else:  # EMERGENCY
            message = (
                f"🚨🚨🚨 FORCED TARGET SWITCH 🚨🚨🚨\n\n"
                f"You MUST now measure: **{target_name}**\n\n"
                f"Design principle:\n{design_principle[:400]}\n\n"
                f"⛔ ABSOLUTELY FORBIDDEN:\n"
                f"  • Do NOT output text\n"
                f"  • Do NOT explain anything\n"
                f"  • Just CALL the tool NOW!"
            )
            
        return {
            "message": message,
            "token_weight": weight,
            "level": level.value
        }
    
    def build_stall_recovery_guidance(self, target_name: str,
                                       design_principle: str,
                                       consecutive_stalls: int) -> dict:
        """
        构建停滞恢复引导消息（包含代码骨架）
        
        Args:
            target_name: 目标名称
            design_principle: 设计原则
            consecutive_stalls: 连续停滞次数
            
        Returns:
            引导消息字典
        """
        weight = 150  # Emergency level always
        
        minimal_skeleton = (
            f"📝 MINIMAL CODE SKELETON for '{target_name}':\n"
            f"```cuda\n"
            f"#include <cuda_runtime.h>\n"
            f"#include <cstdio>\n\n"
            f"__global__ void measure_{target_name}(int* result) {{\n"
            f"    *result = 0; // TODO: Implement measurement\n"
            f"}}\n\n"
            f"int main() {{\n"
            f"    // Allocate, launch kernel, print result\n"
            f"}}\n```\n"
        )
        
        message = (
            f"🚨🚨🚨 STALL RECOVERY ({consecutive_stalls} turns) 🚨🚨🚨\n\n"
            f"{minimal_skeleton}\n\n"
            f"YOUR ONLY TASK: Measure '{target_name}'\n\n"
            f"1. Copy skeleton above\n"
            f"2. Fill in measurement logic\n"
            f"3. Call compile_cuda\n"
            f"4. Call execute_binary\n\n"
            f"Design principle: {design_principle[:300]}"
        )
        
        return {
            "message": message,
            "token_weight": weight,
            "level": "emergency",
            "includes_skeleton": True
        }
    
    def should_inject_completion_hint(self) -> bool:
        """
        判断是否应该注入完成提示
        
        当所有目标都已完成时返回True
        """
        completed, total = self._target_progress
        return total > 0 and completed >= total