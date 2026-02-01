"""
Experience Replay Buffer для DQN с Context7 enterprise паттернами.

Реализует эффективный circular buffer для хранения experience transitions:
- Memory-efficient circular buffer implementation
- Thread-safe operations для production использования
- Batch sampling с configurable batch size
- Optional data preprocessing и normalization
- Comprehensive monitoring и statistics
- Efficient memory management с automatic cleanup
"""

import logging
from typing import List, Optional, Union, Any, Dict, Tuple
from collections import deque, namedtuple
import numpy as np
import torch
import random
import threading
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import pickle
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

# Experience structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBufferConfig(BaseModel):
    """Конфигурация Replay Buffer с валидацией."""
    
    capacity: int = Field(default=100000, description="Максимальный размер buffer", gt=0, le=1000000)
    batch_size: int = Field(default=64, description="Размер batch для sampling", gt=0, le=1024)
    min_size: int = Field(default=1000, description="Минимальный размер перед sampling", gt=0)
    
    # Memory management
    auto_cleanup: bool = Field(default=True, description="Автоматическая очистка памяти")
    cleanup_threshold: float = Field(default=0.9, description="Порог для cleanup", ge=0.5, le=1.0)
    
    # Data preprocessing
    normalize_states: bool = Field(default=False, description="Нормализация состояний")
    normalize_rewards: bool = Field(default=False, description="Нормализация наград")
    reward_scaling: float = Field(default=1.0, description="Масштабирование наград", gt=0)
    
    # Performance
    pin_memory: bool = Field(default=True, description="Pin memory для GPU transfer")
    prefetch_batches: int = Field(default=0, description="Количество prefetch batches", ge=0, le=10)
    
    # Threading
    thread_safe: bool = Field(default=True, description="Thread-safe операции")
    
    @validator("min_size")
    def validate_min_size(cls, v, values):
        if "batch_size" in values and v < values["batch_size"]:
            raise ValueError("min_size должен быть >= batch_size")
        if "capacity" in values and v > values["capacity"]:
            raise ValueError("min_size должен быть <= capacity")
        return v


@dataclass
class BufferStatistics:
    """Статистики Replay Buffer."""
    size: int
    capacity: int
    utilization: float
    total_added: int
    total_sampled: int
    avg_reward: float
    reward_std: float
    memory_usage_mb: float


class ReplayBuffer:
    """
    Experience Replay Buffer с enterprise-grade функциональностью.
    
    Features:
    - Efficient circular buffer с O(1) operations
    - Thread-safe operations для concurrent access
    - Batch sampling с optional preprocessing
    - Memory management и automatic cleanup
    - Comprehensive statistics и monitoring
    - GPU-optimized data transfer
    - Configurable normalization и scaling
    - Persistence support для checkpointing
    """
    
    def __init__(self, 
                 capacity: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 config: Optional[ReplayBufferConfig] = None,
                 device: Union[str, torch.device] = "cpu"):
        """
        Инициализация Replay Buffer.
        
        Args:
            capacity: Максимальный размер buffer (deprecated, используйте config)
            batch_size: Размер batch (deprecated, используйте config)  
            config: Полная конфигурация buffer
            device: Устройство для tensor операций
        """
        # Backward compatibility
        if config is None:
            config = ReplayBufferConfig(
                capacity=capacity or 100000,
                batch_size=batch_size or 64
            )
        
        self.config = config
        self.device = torch.device(device)
        
        # Initialize storage
        self.buffer = deque(maxlen=config.capacity)
        self.position = 0
        self.size = 0
        self.total_added = 0
        self.total_sampled = 0
        
        # Threading support
        if config.thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None
        
        # Statistics tracking
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        
        # Normalization statistics
        self.state_mean = None
        self.state_std = None
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Prefetching
        self._prefetch_cache = []
        
        self.logger = structlog.get_logger(__name__).bind(
            component="ReplayBuffer",
            capacity=config.capacity,
            device=str(self.device)
        )
        
        self.logger.info("Replay Buffer инициализирован", config=config.dict())
    
    def _safe_operation(self, func, *args, **kwargs):
        """Thread-safe wrapper для операций."""
        if self._lock is not None:
            with self._lock:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool) -> None:
        """
        Добавление experience в buffer.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
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
            
            # Добавление в buffer
            self.buffer.append(experience)
            
            # Обновление статистик
            self.total_added += 1
            self.reward_sum += reward
            self.reward_sq_sum += reward ** 2
            
            # Обновление размера
            self.size = len(self.buffer)
            
            # Обновление нормализации
            if self.config.normalize_states:
                self._update_state_normalization(state)
            
            if self.config.normalize_rewards:
                self._update_reward_normalization()
            
            # Автоматическая очистка если нужна
            if (self.config.auto_cleanup and 
                self.size / self.config.capacity > self.config.cleanup_threshold):
                self._cleanup_memory()
        
        self._safe_operation(_push)
        
        if self.total_added % 10000 == 0:
            self.logger.debug("Buffer status", 
                            size=self.size, 
                            utilization=self.size/self.config.capacity)
    
    def sample(self, batch_size: Optional[int] = None) -> List[Experience]:
        """
        Sampling random batch из buffer.
        
        Args:
            batch_size: Размер batch (по умолчанию из config)
            
        Returns:
            Список experiences
        """
        def _sample():
            if self.size < self.config.min_size:
                raise ValueError(f"Недостаточно данных в buffer: {self.size} < {self.config.min_size}")
            
            effective_batch_size = batch_size or self.config.batch_size
            
            # Uniform sampling
            batch_indices = random.sample(range(self.size), effective_batch_size)
            batch = [self.buffer[i] for i in batch_indices]
            
            # Preprocessing если включен
            if self.config.normalize_states or self.config.normalize_rewards:
                batch = self._preprocess_batch(batch)
            
            self.total_sampled += effective_batch_size
            
            return batch
        
        return self._safe_operation(_sample)
    
    def sample_tensors(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sampling batch и конвертация в tensors.
        
        Args:
            batch_size: Размер batch
            
        Returns:
            Tuple из (states, actions, rewards, next_states, dones)
        """
        batch = self.sample(batch_size)
        
        # Конвертация в numpy arrays
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Конвертация в tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Pin memory для faster GPU transfer если включено
        if self.config.pin_memory and self.device.type == 'cuda':
            states_tensor = states_tensor.pin_memory()
            actions_tensor = actions_tensor.pin_memory()
            rewards_tensor = rewards_tensor.pin_memory()
            next_states_tensor = next_states_tensor.pin_memory()
            dones_tensor = dones_tensor.pin_memory()
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def _preprocess_batch(self, batch: List[Experience]) -> List[Experience]:
        """
        Preprocessing batch данных.
        
        Args:
            batch: Исходный batch
            
        Returns:
            Обработанный batch
        """
        processed_batch = []
        
        for experience in batch:
            state = experience.state
            next_state = experience.next_state
            reward = experience.reward
            
            # Нормализация состояний
            if self.config.normalize_states and self.state_std is not None:
                state = (state - self.state_mean) / (self.state_std + 1e-8)
                next_state = (next_state - self.state_mean) / (self.state_std + 1e-8)
            
            # Нормализация наград
            if self.config.normalize_rewards:
                reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
            
            # Масштабирование наград
            if self.config.reward_scaling != 1.0:
                reward = reward * self.config.reward_scaling
            
            processed_experience = Experience(
                state=state,
                action=experience.action,
                reward=reward,
                next_state=next_state,
                done=experience.done
            )
            processed_batch.append(processed_experience)
        
        return processed_batch
    
    def _update_state_normalization(self, state: np.ndarray) -> None:
        """Обновление статистик нормализации состояний."""
        if self.state_mean is None:
            self.state_mean = state.copy()
            self.state_std = np.zeros_like(state)
        else:
            # Running mean и std
            alpha = 1.0 / min(1000, self.total_added)  # Decay factor
            delta = state - self.state_mean
            self.state_mean += alpha * delta
            self.state_std = (1 - alpha) * self.state_std + alpha * (delta ** 2)
    
    def _update_reward_normalization(self) -> None:
        """Обновление статистик нормализации наград."""
        if self.total_added > 1:
            self.reward_mean = self.reward_sum / self.total_added
            self.reward_std = np.sqrt(max(
                self.reward_sq_sum / self.total_added - self.reward_mean ** 2,
                1e-8
            ))
    
    def _cleanup_memory(self) -> None:
        """Очистка памяти при переполнении."""
        # Принудительная сборка мусора может помочь
        import gc
        gc.collect()
        
        self.logger.debug("Memory cleanup выполнен")
    
    def clear(self) -> None:
        """Полная очистка buffer."""
        def _clear():
            self.buffer.clear()
            self.size = 0
            self.position = 0
            self.reward_sum = 0.0
            self.reward_sq_sum = 0.0
            self.state_mean = None
            self.state_std = None
            self.reward_mean = 0.0
            self.reward_std = 1.0
        
        self._safe_operation(_clear)
        self.logger.info("Buffer очищен")
    
    def get_statistics(self) -> BufferStatistics:
        """
        Получить статистики buffer.
        
        Returns:
            Статистики buffer
        """
        def _get_stats():
            avg_reward = self.reward_mean if self.total_added > 0 else 0.0
            reward_std = self.reward_std if self.total_added > 1 else 0.0
            
            # Примерная оценка использования памяти
            memory_usage = 0.0
            if self.size > 0:
                sample_experience = self.buffer[0]
                state_size = sample_experience.state.nbytes
                next_state_size = sample_experience.next_state.nbytes
                experience_size = state_size + next_state_size + 16  # action, reward, done
                memory_usage = (experience_size * self.size) / (1024 * 1024)  # MB
            
            return BufferStatistics(
                size=self.size,
                capacity=self.config.capacity,
                utilization=self.size / self.config.capacity,
                total_added=self.total_added,
                total_sampled=self.total_sampled,
                avg_reward=avg_reward,
                reward_std=reward_std,
                memory_usage_mb=memory_usage
            )
        
        return self._safe_operation(_get_stats)
    
    def save(self, filepath: str) -> None:
        """
        Сохранение buffer в файл.
        
        Args:
            filepath: Путь для сохранения
        """
        def _save():
            save_data = {
                'config': self.config.dict(),
                'buffer': list(self.buffer),
                'position': self.position,
                'size': self.size,
                'total_added': self.total_added,
                'total_sampled': self.total_sampled,
                'reward_sum': self.reward_sum,
                'reward_sq_sum': self.reward_sq_sum,
                'state_mean': self.state_mean,
                'state_std': self.state_std,
                'reward_mean': self.reward_mean,
                'reward_std': self.reward_std,
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        
        self._safe_operation(_save)
        self.logger.info("Buffer сохранен", filepath=filepath)
    
    @classmethod
    def load(cls, filepath: str, device: Union[str, torch.device] = "cpu") -> 'ReplayBuffer':
        """
        Загрузка buffer из файла.
        
        Args:
            filepath: Путь к файлу
            device: Устройство для operations
            
        Returns:
            Загруженный buffer
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Создание buffer с сохраненной конфигурацией
        config = ReplayBufferConfig(**save_data['config'])
        buffer = cls(config=config, device=device)
        
        # Восстановление состояния
        buffer.buffer = deque(save_data['buffer'], maxlen=config.capacity)
        buffer.position = save_data['position']
        buffer.size = save_data['size']
        buffer.total_added = save_data['total_added']
        buffer.total_sampled = save_data['total_sampled']
        buffer.reward_sum = save_data['reward_sum']
        buffer.reward_sq_sum = save_data['reward_sq_sum']
        buffer.state_mean = save_data['state_mean']
        buffer.state_std = save_data['state_std']
        buffer.reward_mean = save_data['reward_mean']
        buffer.reward_std = save_data['reward_std']
        
        buffer.logger.info("Buffer загружен", filepath=filepath, size=buffer.size)
        return buffer
    
    def __len__(self) -> int:
        """Размер buffer."""
        return self.size
    
    def __repr__(self) -> str:
        """Строковое представление buffer."""
        return (
            f"ReplayBuffer(size={self.size}, capacity={self.config.capacity}, "
            f"utilization={self.size/self.config.capacity:.1%})"
        )