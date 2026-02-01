"""
Prioritized Experience Replay Buffer с Context7 enterprise паттернами.

Реализует PER (Prioritized Experience Replay) с sum-tree для эффективного sampling:
- Sum-tree data structure для O(log n) sampling  
- TD-error based priorities для better learning
- Importance sampling weights для bias correction
- Configurable alpha и beta parameters
- Production-ready thread safety
- Memory-efficient implementation
- Comprehensive monitoring
"""

import logging
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np
import torch
import threading
import random
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import structlog
from collections import namedtuple

from .replay_buffer import Experience, ReplayBuffer, ReplayBufferConfig

logger = structlog.get_logger(__name__)


class PrioritizedReplayConfig(ReplayBufferConfig):
    """Конфигурация Prioritized Experience Replay с валидацией."""
    
    # PER specific parameters
    alpha: float = Field(default=0.6, description="Priority exponent", ge=0, le=1.0)
    beta_start: float = Field(default=0.4, description="Initial importance sampling exponent", ge=0, le=1.0)
    beta_end: float = Field(default=1.0, description="Final importance sampling exponent", ge=0, le=1.0)
    beta_frames: int = Field(default=100000, description="Frames to anneal beta", gt=0)
    
    # Priority management
    max_priority: float = Field(default=1.0, description="Maximum priority value", gt=0)
    min_priority: float = Field(default=1e-6, description="Minimum priority value", gt=0, le=1e-3)
    priority_epsilon: float = Field(default=1e-6, description="Small epsilon for numerical stability", gt=0)
    
    # Performance optimization
    tree_auto_rebalance: bool = Field(default=True, description="Auto rebalance sum tree")
    rebalance_threshold: int = Field(default=10000, description="Operations before rebalance", gt=0)
    
    @validator("beta_start")
    def validate_beta_start(cls, v, values):
        if "beta_end" in values and v > values["beta_end"]:
            raise ValueError("beta_start должен быть <= beta_end")
        return v
    
    @validator("min_priority")
    def validate_min_priority(cls, v, values):
        if "max_priority" in values and v >= values["max_priority"]:
            raise ValueError("min_priority должен быть < max_priority")
        return v


@dataclass
class PrioritizedExperience:
    """Experience с priority информацией."""
    experience: Experience
    priority: float
    tree_index: int


class SumTree:
    """
    Sum Tree data structure для эффективного prioritized sampling.
    
    Бинарное дерево где:
    - Листья содержат priorities
    - Внутренние узлы содержат суммы потомков
    - Root содержит общую сумму всех priorities
    
    Операции:
    - Update: O(log n)
    - Sample: O(log n) 
    - Total sum: O(1)
    """
    
    def __init__(self, capacity: int):
        """
        Инициализация Sum Tree.
        
        Args:
            capacity: Максимальное количество элементов
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.full(capacity, None)
        self.data_pointer = 0
        self.size = 0
        
        # Statistics для monitoring
        self.total_updates = 0
        self.total_samples = 0
        
        self.logger = structlog.get_logger(__name__).bind(component="SumTree")
        self.logger.debug("SumTree инициализировано", capacity=capacity)
    
    def add(self, priority: float, data: Any) -> int:
        """
        Добавление элемента с priority.
        
        Args:
            priority: Priority value
            data: Data для хранения
            
        Returns:
            Tree index добавленного элемента
        """
        tree_index = self.data_pointer + self.capacity - 1
        
        # Сохранение data
        self.data[self.data_pointer] = data
        
        # Обновление tree с новым priority
        self.update(tree_index, priority)
        
        # Циклический указатель
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Увеличение размера до максимума
        if self.size < self.capacity:
            self.size += 1
        
        return tree_index
    
    def update(self, tree_index: int, priority: float) -> None:
        """
        Обновление priority элемента.
        
        Args:
            tree_index: Индекс в дереве
            priority: Новый priority
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Обновление всех parent узлов
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
        
        self.total_updates += 1
    
    def get_leaf(self, value: float) -> Tuple[int, float, Any]:
        """
        Получение leaf node по cumulative value.
        
        Args:
            value: Cumulative value для поиска
            
        Returns:
            Tuple из (leaf_index, priority, data)
        """
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # Если достигли leaf
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            # Выбираем left или right child
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        priority = self.tree[leaf_index]
        data = self.data[data_index]
        
        self.total_samples += 1
        
        return leaf_index, priority, data
    
    def total_priority(self) -> float:
        """Получить общую сумму priorities."""
        return self.tree[0]
    
    def max_priority(self) -> float:
        """Получить максимальный priority."""
        if self.size == 0:
            return 1.0
        
        # Максимальный priority среди используемых листьев
        start_idx = self.capacity - 1
        end_idx = start_idx + self.size
        return np.max(self.tree[start_idx:end_idx])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистики Sum Tree."""
        return {
            "capacity": self.capacity,
            "size": self.size,
            "total_priority": self.total_priority(),
            "max_priority": self.max_priority(),
            "avg_priority": self.total_priority() / max(self.size, 1),
            "total_updates": self.total_updates,
            "total_samples": self.total_samples,
            "utilization": self.size / self.capacity,
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer с enterprise функциональностью.
    
    Features:
    - Sum-tree based prioritized sampling для efficient O(log n) operations
    - TD-error based priorities для better learning efficiency
    - Importance sampling weights для bias correction
    - Configurable alpha/beta parameters с annealing
    - Thread-safe operations для production использования
    - Memory-efficient implementation с auto-rebalancing
    - Comprehensive monitoring и statistics
    - Adaptive priority management
    """
    
    def __init__(self,
                 capacity: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 config: Optional[PrioritizedReplayConfig] = None,
                 device: Union[str, torch.device] = "cpu"):
        """
        Инициализация Prioritized Replay Buffer.
        
        Args:
            capacity: Максимальный размер buffer (deprecated)
            batch_size: Размер batch (deprecated)
            config: Полная конфигурация buffer
            device: Устройство для tensor операций
        """
        # Backward compatibility
        if config is None:
            config = PrioritizedReplayConfig(
                capacity=capacity or 100000,
                batch_size=batch_size or 64
            )
        
        super().__init__(config=config, device=device)
        
        self.per_config = config
        
        # Initialize sum tree
        self.sum_tree = SumTree(config.capacity)
        
        # Priority management
        self.max_priority = config.max_priority
        self.priority_updates = 0
        
        # Beta annealing для importance sampling
        self.beta_frames_completed = 0
        
        # Performance tracking
        self.rebalance_operations = 0
        
        self.logger = structlog.get_logger(__name__).bind(
            component="PrioritizedReplayBuffer",
            capacity=config.capacity,
            alpha=config.alpha,
            device=str(self.device)
        )
        
        self.logger.info("Prioritized Replay Buffer инициализирован")
    
    def get_beta(self) -> float:
        """
        Получить текущий beta для importance sampling.
        
        Returns:
            Current beta value
        """
        if self.beta_frames_completed >= self.per_config.beta_frames:
            return self.per_config.beta_end
        
        # Linear annealing
        progress = self.beta_frames_completed / self.per_config.beta_frames
        beta = self.per_config.beta_start + progress * (
            self.per_config.beta_end - self.per_config.beta_start
        )
        
        return beta
    
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool,
             priority: Optional[float] = None) -> None:
        """
        Добавление experience с priority.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние  
            done: Флаг завершения эпизода
            priority: Priority для experience (если None, используется max)
        """
        def _push():
            # Создание experience
            experience = Experience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done
            )
            
            # Определение priority
            if priority is None:
                priority = self.max_priority
            
            # Clamp priority в допустимый диапазон
            priority = np.clip(priority, self.per_config.min_priority, self.per_config.max_priority)
            
            # Добавление в sum tree
            tree_index = self.sum_tree.add(priority, experience)
            
            # Обновление базовой статистики
            self.total_added += 1
            self.reward_sum += reward
            self.reward_sq_sum += reward ** 2
            self.size = min(self.total_added, self.per_config.capacity)
            
            # Обновление максимального priority
            if priority > self.max_priority:
                self.max_priority = priority
            
            # Auto-rebalancing если включено
            if (self.per_config.tree_auto_rebalance and 
                self.rebalance_operations % self.per_config.rebalance_threshold == 0):
                self._rebalance_tree()
            
            self.rebalance_operations += 1
        
        self._safe_operation(_push)
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Prioritized sampling из buffer.
        
        Args:
            batch_size: Размер batch
            
        Returns:
            Tuple из (experiences, importance_weights, tree_indices)
        """
        def _sample():
            if self.size < self.per_config.min_size:
                raise ValueError(f"Недостаточно данных: {self.size} < {self.per_config.min_size}")
            
            effective_batch_size = batch_size or self.per_config.batch_size
            
            # Sampling с priorities
            experiences = []
            tree_indices = []
            priorities = []
            
            total_priority = self.sum_tree.total_priority()
            segment_size = total_priority / effective_batch_size
            
            for i in range(effective_batch_size):
                # Uniform sampling в каждом segment
                segment_start = segment_size * i
                segment_end = segment_start + segment_size
                sample_value = np.random.uniform(segment_start, segment_end)
                
                # Получение experience из sum tree
                tree_index, priority, experience = self.sum_tree.get_leaf(sample_value)
                
                experiences.append(experience)
                tree_indices.append(tree_index)
                priorities.append(priority)
            
            # Вычисление importance sampling weights
            priorities = np.array(priorities, dtype=np.float32)
            sampling_probs = priorities / total_priority
            
            # IS weights с current beta
            beta = self.get_beta()
            is_weights = (self.size * sampling_probs) ** (-beta)
            is_weights = is_weights / np.max(is_weights)  # Normalize
            
            # Обновление статистики
            self.total_sampled += effective_batch_size
            self.beta_frames_completed += effective_batch_size
            
            return experiences, is_weights, np.array(tree_indices)
        
        return self._safe_operation(_sample)
    
    def sample_tensors(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Prioritized sampling с конвертацией в tensors.
        
        Args:
            batch_size: Размер batch
            
        Returns:
            Tuple из (states, actions, rewards, next_states, dones, is_weights, tree_indices)
        """
        experiences, is_weights, tree_indices = self.sample(batch_size)
        
        # Конвертация experiences в tensors (как в базовом классе)
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Создание tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)
        tree_indices_tensor = torch.LongTensor(tree_indices).to(self.device)
        
        return (states_tensor, actions_tensor, rewards_tensor, 
                next_states_tensor, dones_tensor, is_weights_tensor, tree_indices_tensor)
    
    def update_priorities(self, tree_indices: Union[List[int], np.ndarray], 
                         td_errors: Union[List[float], np.ndarray]) -> None:
        """
        Обновление priorities на основе TD errors.
        
        Args:
            tree_indices: Indices в sum tree
            td_errors: TD errors для соответствующих experiences
        """
        def _update():
            if isinstance(tree_indices, torch.Tensor):
                tree_indices_np = tree_indices.cpu().numpy()
            else:
                tree_indices_np = np.array(tree_indices)
                
            if isinstance(td_errors, torch.Tensor):
                td_errors_np = td_errors.cpu().numpy()
            else:
                td_errors_np = np.array(td_errors)
            
            # Вычисление новых priorities
            priorities = (np.abs(td_errors_np) + self.per_config.priority_epsilon) ** self.per_config.alpha
            
            # Clamp priorities
            priorities = np.clip(priorities, self.per_config.min_priority, self.per_config.max_priority)
            
            # Обновление в sum tree
            for tree_index, priority in zip(tree_indices_np, priorities):
                self.sum_tree.update(tree_index, priority)
                
                # Обновление максимального priority
                if priority > self.max_priority:
                    self.max_priority = priority
            
            self.priority_updates += len(tree_indices_np)
        
        self._safe_operation(_update)
    
    def _rebalance_tree(self) -> None:
        """Rebalancing sum tree для оптимальной производительности."""
        # Простая реализация - периодический пересчет статистик
        self.max_priority = self.sum_tree.max_priority()
        
        self.logger.debug("Sum tree rebalanced", 
                         max_priority=self.max_priority,
                         total_priority=self.sum_tree.total_priority())
    
    def get_priority_statistics(self) -> Dict[str, Any]:
        """Получить статистики priorities."""
        tree_stats = self.sum_tree.get_statistics()
        
        return {
            "max_priority": self.max_priority,
            "priority_updates": self.priority_updates,
            "current_beta": self.get_beta(),
            "beta_progress": min(self.beta_frames_completed / self.per_config.beta_frames, 1.0),
            "tree_statistics": tree_stats,
            "alpha": self.per_config.alpha,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Расширенные статистики для PER buffer."""
        base_stats = super().get_statistics()
        priority_stats = self.get_priority_statistics()
        
        # Combine статистики
        stats_dict = base_stats.__dict__.copy()
        stats_dict.update({
            "priority_stats": priority_stats,
            "buffer_type": "PrioritizedReplayBuffer"
        })
        
        return stats_dict
    
    def clear(self) -> None:
        """Полная очистка prioritized buffer."""
        super().clear()
        
        # Очистка PER-специфичных компонентов
        self.sum_tree = SumTree(self.per_config.capacity)
        self.max_priority = self.per_config.max_priority
        self.priority_updates = 0
        self.beta_frames_completed = 0
        self.rebalance_operations = 0
        
        self.logger.info("Prioritized buffer очищен")
    
    def __repr__(self) -> str:
        """Строковое представление prioritized buffer."""
        return (
            f"PrioritizedReplayBuffer(size={self.size}, capacity={self.per_config.capacity}, "
            f"alpha={self.per_config.alpha}, beta={self.get_beta():.3f}, "
            f"utilization={self.size/self.per_config.capacity:.1%})"
        )