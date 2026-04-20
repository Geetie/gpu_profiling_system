"""
TargetStateMachine - 统一目标状态管理器

解决AgentLoop God Object问题，将目标状态逻辑从agent_loop.py中分离。
符合spec.md §3.2双层代理循环设计原则。

职责:
- 维护完整的目标列表 (all_targets)
- 追踪已完成/未完成/失败的目标
- 提供清晰的状态转换API
- 支持事件通知机制
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetState:
    """不可变的目标状态快照"""
    all_targets: tuple[str, ...]
    completed_targets: tuple[str, ...]
    current_target: Optional[str]
    total_count: int
    completed_count: int
    unmeasured_count: int


class TargetStateMachine:
    """
    统一目标状态机
    
    设计原则:
    - P4 可组合性: 可独立使用或嵌入其他组件
    - P6 状态落盘: 支持序列化/反序列化
    - 单一职责: 只负责目标状态管理，不涉及LLM交互
    
    状态转换图:
    
    ┌─────────┐   start_target()   ┌───────────┐
    │ INITIAL │ ────────────────> │ MEASURING │
    └─────────┘                   └───────────┘
                                      │
                              complete_target()
                                      ↓
                                ┌───────────┐
                                │ COMPLETED │
                                └───────────┘
                                      │
                                  fail_target() (retry limit)
                                      ↓
                                ┌───────────┐
                                │ FAILED    │
                                └───────────┘
    """
    
    def __init__(self):
        self._all_targets: list[str] = []
        self._completed_targets: list[str] = []
        self._current_target: Optional[str] = None
        self._target_retry_count: dict[str, int] = {}
        self._max_retries_per_target: int = 3
        self._state_change_callbacks: list[Callable[[TargetState], None]] = []
        
    @property
    def is_initialized(self) -> bool:
        return len(self._all_targets) > 0
    
    @property
    def current_target(self) -> Optional[str]:
        return self._current_target
    
    @property
    def completed_targets(self) -> list[str]:
        return list(self._completed_targets)
    
    @property
    def unmeasured_targets(self) -> list[str]:
        """返回未测量的目标列表"""
        return [t for t in self._all_targets if t not in self._completed_targets]
    
    @property
    def is_all_completed(self) -> bool:
        """检查是否所有目标都已完成"""
        return len(self._all_targets) > 0 and len(self.unmeasured_targets) == 0
    
    @property
    def progress(self) -> tuple[int, int]:
        """返回进度 (完成数, 总数)"""
        return len(self._completed_targets), len(self._all_targets)
    
    def initialize(self, targets: list[str]) -> None:
        """
        初始化目标列表
        
        Args:
            targets: 完整的目标列表 (如 ['dram_latency_cycles', 'l2_cache_size_mb'])
        
        Raises:
            ValueError: 如果targets为空或已初始化
        """
        if not targets:
            raise ValueError("Targets list cannot be empty")
        if self.is_initialized:
            logger.warning("TargetStateMachine already initialized, reinitializing...")
            
        self._all_targets = list(targets)
        self._completed_targets = []
        self._current_target = None
        self._target_retry_count = {t: 0 for t in targets}
        
        logger.info(f"TargetStateMachine initialized with {len(targets)} targets: {targets}")
        self._notify_state_change()
    
    def start_first_target(self) -> str:
        """
        开始第一个目标的测量
        
        Returns:
            第一个目标名称
            
        Raises:
            RuntimeError: 如果未初始化或无可用目标
        """
        if not self.is_initialized:
            raise RuntimeError("TargetStateMachine not initialized")
            
        next_target = self.unmeasured_targets[0]
        self._current_target = next_target
        
        logger.info(f"Starting first target: {next_target}")
        self._notify_state_change()
        return next_target
    
    def complete_current_target(self) -> Optional[str]:
        """
        标记当前目标为已完成，自动切换到下一目标
        
        Returns:
            下一个目标名称，如果全部完成则返回None
        """
        if not self._current_target:
            logger.warning("No current target to complete")
            return None
            
        # 标记为完成
        if self._current_target not in self._completed_targets:
            self._completed_targets.append(self._current_target)
            logger.info(f"✅ Target completed: {self._current_target} "
                       f"({len(self._completed_targets)}/{len(self._all_targets)})")
        
        # 切换到下一个目标
        next_target = self._advance_to_next_target()
        self._notify_state_change()
        
        return next_target
    
    def fail_current_target(self) -> Optional[str]:
        """
        标记当前目标为失败(达到重试上限)，切换到下一目标
        
        Returns:
            下一个目标名称，如果全部完成则返回None
        """
        if not self._current_target:
            return None
            
        target = self._current_target
        self._target_retry_count[target] += 1
        
        if self._target_retry_count[target] >= self._max_retries_per_target:
            logger.warning(f"⚠️ Target failed after {self._max_retries_per_target} retries: {target}")
            # 强制标记为完成以避免死循环
            if target not in self._completed_targets:
                self._completed_targets.append(target)
                
        next_target = self._advance_to_next_target()
        self._notify_state_change()
        
        return next_target
    
    def force_switch_to_target(self, target_name: str) -> bool:
        """
        强制切换到指定目标（用于BUG#2强制恢复机制）
        
        Args:
            target_name: 目标名称
            
        Returns:
            是否成功切换
        """
        if target_name not in self._all_targets:
            logger.error(f"Invalid target name: {target_name}")
            return False
            
        if target_name in self._completed_targets:
            logger.warning(f"Target already completed: {target_name}")
            
        prev_target = self._current_target
        self._current_target = target_name
        self._target_retry_count[target_name] = 0
        
        logger.info(f"Force-switched: {prev_target} → {target_name}")
        self._notify_state_change()
        return True
    
    def get_snapshot(self) -> TargetState:
        """返回当前状态的不可变快照"""
        return TargetState(
            all_targets=tuple(self._all_targets),
            completed_targets=tuple(self._completed_targets),
            current_target=self._current_target,
            total_count=len(self._all_targets),
            completed_count=len(self._completed_targets),
            unmeasured_count=len(self.unmeasured_targets)
        )
    
    def to_dict(self) -> dict:
        """序列化为字典 (P6: 状态落盘)"""
        return {
            "all_targets": self._all_targets,
            "completed_targets": self._completed_targets,
            "current_target": self._current_target,
            "target_retry_count": dict(self._target_retry_count),
            "is_initialized": self.is_initialized,
            "progress": {
                "completed": len(self._completed_targets),
                "total": len(self._all_targets)
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TargetStateMachine':
        """从字典反序列化 (P6: 状态恢复)"""
        machine = cls()
        machine._all_targets = data.get("all_targets", [])
        machine._completed_targets = data.get("completed_targets", [])
        machine._current_target = data.get("current_target")
        machine._target_retry_count = data.get("target_retry_count", {})
        return machine
    
    def on_state_change(self, callback: Callable[[TargetState], None]) -> None:
        """
        注册状态变更回调 (用于事件通知)
        
        Args:
            callback: 接受TargetState参数的回调函数
        """
        self._state_change_callbacks.append(callback)
    
    def _advance_to_next_target(self) -> Optional[str]:
        """内部方法：推进到下一个未测量目标"""
        unmeasured = self.unmeasured_targets
        if not unmeasured:
            self._current_target = None
            logger.info("✅ All targets measured!")
            return None
            
        next_target = unmeasured[0]
        self._current_target = next_target
        logger.info(f"Switching to next target: {next_target}")
        return next_target
    
    def _notify_state_change(self) -> None:
        """通知所有注册的状态变更监听器"""
        snapshot = self.get_snapshot()
        for callback in self._state_change_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def __repr__(self) -> str:
        completed, total = self.progress
        return (f"TargetStateMachine("
                f"total={total}, "
                f"completed={completed}, "
                f"current='{self._current_target}', "
                f"unmeasured={self.unmeasured_targets})")