"""
Deep Q-Network (DQN) implementation с Context7 enterprise паттернами.

Полная реализация DQN алгоритма с современными оптимизациями:
- Epsilon-greedy exploration с адаптивным decay
- Target network для стабильности обучения
- Experience replay для эффективного использования данных
- Gradient clipping и regularization
- Comprehensive monitoring и logging
- Production-ready error handling
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, Field, validator
import structlog
from dataclasses import dataclass
from datetime import datetime
import pickle
import json

from ..networks.q_network import QNetwork, QNetworkConfig
from ..buffers.replay_buffer import ReplayBuffer
from ..utils.epsilon_schedule import EpsilonSchedule

logger = structlog.get_logger(__name__)


@dataclass
class Experience:
    """Структура для хранения опыта (s, a, r, s', done)."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации."""
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
        }


class DQNConfig(BaseModel):
    """Конфигурация DQN агента с валидацией."""
    
    # Network configuration
    network_config: QNetworkConfig = Field(..., description="Конфигурация Q-network")
    
    # Training hyperparameters
    learning_rate: float = Field(default=1e-4, description="Learning rate", gt=0, le=1e-1)
    gamma: float = Field(default=0.99, description="Discount factor", ge=0, le=1.0)
    epsilon_start: float = Field(default=1.0, description="Начальный epsilon", ge=0, le=1.0)
    epsilon_end: float = Field(default=0.01, description="Конечный epsilon", ge=0, le=1.0)
    epsilon_decay: float = Field(default=0.995, description="Decay rate epsilon", ge=0.9, le=1.0)
    
    # Experience replay
    buffer_size: int = Field(default=100000, description="Размер replay buffer", gt=0)
    batch_size: int = Field(default=64, description="Размер batch", gt=0, le=512)
    min_replay_size: int = Field(default=1000, description="Минимум опыта для обучения", gt=0)
    
    # Target network updates
    target_update_freq: int = Field(default=1000, description="Частота обновления target network", gt=0)
    soft_update_tau: Optional[float] = Field(default=None, description="Tau для soft updates", ge=0, le=1.0)
    
    # Optimization
    optimizer_type: str = Field(default="adam", description="Тип оптимизатора")
    weight_decay: float = Field(default=1e-5, description="L2 regularization", ge=0)
    grad_clip_norm: float = Field(default=1.0, description="Gradient clipping norm", gt=0)
    
    # Loss function
    loss_type: str = Field(default="mse", description="Тип loss function")
    huber_delta: float = Field(default=1.0, description="Delta для Huber loss", gt=0)
    
    # Monitoring
    log_freq: int = Field(default=1000, description="Частота логирования", gt=0)
    save_freq: int = Field(default=10000, description="Частота сохранения", gt=0)
    eval_freq: int = Field(default=5000, description="Частота evaluation", gt=0)
    
    # Device
    device: str = Field(default="auto", description="Устройство для вычислений")
    seed: Optional[int] = Field(default=None, description="Random seed")
    
    @validator("epsilon_end")
    def validate_epsilon_end(cls, v, values):
        if "epsilon_start" in values and v >= values["epsilon_start"]:
            raise ValueError("epsilon_end должен быть меньше epsilon_start")
        return v
    
    @validator("min_replay_size")
    def validate_min_replay_size(cls, v, values):
        if "batch_size" in values and v < values["batch_size"]:
            raise ValueError("min_replay_size должен быть >= batch_size")
        return v
    
    @validator("optimizer_type")
    def validate_optimizer(cls, v):
        valid_optimizers = ["adam", "adamw", "rmsprop", "sgd"]
        if v not in valid_optimizers:
            raise ValueError(f"Оптимизатор должен быть одним из: {valid_optimizers}")
        return v
    
    @validator("loss_type")
    def validate_loss(cls, v):
        valid_losses = ["mse", "huber", "smooth_l1"]
        if v not in valid_losses:
            raise ValueError(f"Loss должен быть одним из: {valid_losses}")
        return v


class DQN:
    """
    Deep Q-Network (DQN) агент с enterprise-grade реализацией.
    
    Features:
    - Epsilon-greedy exploration с адаптивными стратегиями
    - Target network для стабильности обучения
    - Experience replay для эффективного использования данных
    - Configurable loss functions (MSE, Huber, Smooth L1)
    - Gradient clipping и regularization
    - Comprehensive logging и monitoring
    - Checkpointing и model persistence
    - Production-ready error handling
    """
    
    def __init__(self, config: DQNConfig):
        """
        Инициализация DQN агента.
        
        Args:
            config: Конфигурация агента
        """
        self.config = config
        self.training_step = 0
        self.episode_rewards = []
        self.loss_history = []
        
        # Настройка устройства
        self.device = self._setup_device()
        
        # Настройка random seed
        if config.seed is not None:
            self._set_seed(config.seed)
        
        # Инициализация компонентов
        self._initialize_networks()
        self._initialize_optimizer()
        self._initialize_replay_buffer()
        self._initialize_epsilon_schedule()
        
        # Logging setup
        self.logger = structlog.get_logger(__name__).bind(
            agent_type="DQN",
            device=str(self.device)
        )
        
        self.logger.info("DQN агент инициализирован", config=config.dict())
    
    def _setup_device(self) -> torch.device:
        """Настройка устройства для вычислений."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"CUDA доступна: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.info("Используется Apple Metal Performance Shaders (MPS)")
            else:
                device = torch.device("cpu")
                self.logger.info("Используется CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _set_seed(self, seed: int) -> None:
        """Установка random seed для воспроизводимости."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Для полной детерминированности (может замедлить обучение)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.logger.info("Random seed установлен", seed=seed)
    
    def _initialize_networks(self) -> None:
        """Инициализация Q-networks."""
        # Главная сеть
        self.q_network = QNetwork(self.config.network_config).to(self.device)
        
        # Целевая сеть (копия главной)
        self.target_network = QNetwork(self.config.network_config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Заморозка целевой сети
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.logger.info("Q-networks инициализированы", 
                        params=self.q_network.get_network_stats()["total_parameters"])
    
    def _initialize_optimizer(self) -> None:
        """Инициализация оптимизатора."""
        optimizer_map = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "rmsprop": optim.RMSprop,
            "sgd": optim.SGD,
        }
        
        optimizer_class = optimizer_map[self.config.optimizer_type]
        optimizer_kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }
        
        if self.config.optimizer_type in ["adam", "adamw"]:
            optimizer_kwargs.update({"betas": (0.9, 0.999), "eps": 1e-8})
        elif self.config.optimizer_type == "sgd":
            optimizer_kwargs.update({"momentum": 0.9})
        
        self.optimizer = optimizer_class(self.q_network.parameters(), **optimizer_kwargs)
        
        self.logger.info("Оптимизатор инициализирован", 
                        type=self.config.optimizer_type, 
                        lr=self.config.learning_rate)
    
    def _initialize_replay_buffer(self) -> None:
        """Инициализация replay buffer."""
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            batch_size=self.config.batch_size,
            device=self.device
        )
        
        self.logger.info("Replay buffer инициализирован", capacity=self.config.buffer_size)
    
    def _initialize_epsilon_schedule(self) -> None:
        """Инициализация расписания epsilon."""
        self.epsilon_schedule = EpsilonSchedule(
            start_epsilon=self.config.epsilon_start,
            end_epsilon=self.config.epsilon_end,
            decay_rate=self.config.epsilon_decay
        )
        
        self.logger.info("Epsilon schedule инициализирован")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Выбор действия с epsilon-greedy policy.
        
        Args:
            state: Текущее состояние
            training: Режим обучения (влияет на epsilon)
            
        Returns:
            Выбранное действие
        """
        if training and np.random.random() < self.epsilon_schedule.get_epsilon(self.training_step):
            # Случайное действие (exploration)
            action = np.random.randint(0, self.config.network_config.action_size)
        else:
            # Greedy действие (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        """
        Сохранение опыта в replay buffer.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_step(self) -> Dict[str, float]:
        """
        Один шаг обучения.
        
        Returns:
            Метрики обучения
        """
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {"status": "insufficient_data"}
        
        # Sampling batch из replay buffer
        experiences = self.replay_buffer.sample()
        
        # Конвертация в tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values от target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self._compute_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.grad_clip_norm
        )
        
        self.optimizer.step()
        
        # Update training step
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.config.target_update_freq == 0:
            self._update_target_network()
        
        # Logging
        metrics = {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "epsilon": self.epsilon_schedule.get_epsilon(self.training_step),
            "training_step": self.training_step,
            "q_mean": current_q_values.mean().item(),
            "target_mean": target_q_values.mean().item(),
        }
        
        self.loss_history.append(loss.item())
        
        if self.training_step % self.config.log_freq == 0:
            self.logger.info("Training step completed", **metrics)
        
        return metrics
    
    def _compute_loss(self, current_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """
        Вычисление loss function.
        
        Args:
            current_q: Текущие Q-values
            target_q: Целевые Q-values
            
        Returns:
            Loss value
        """
        if self.config.loss_type == "mse":
            return F.mse_loss(current_q, target_q)
        elif self.config.loss_type == "huber":
            return F.huber_loss(current_q, target_q, delta=self.config.huber_delta)
        elif self.config.loss_type == "smooth_l1":
            return F.smooth_l1_loss(current_q, target_q)
        else:
            raise ValueError(f"Неизвестный тип loss: {self.config.loss_type}")
    
    def _update_target_network(self) -> None:
        """Обновление target network."""
        if self.config.soft_update_tau is not None:
            # Soft update
            self.q_network.soft_update(self.target_network, self.config.soft_update_tau)
        else:
            # Hard update
            self.q_network.hard_update(self.target_network)
        
        self.logger.debug("Target network обновлена", step=self.training_step)
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluation агента в environment.
        
        Args:
            env: Environment для тестирования
            num_episodes: Количество эпизодов для оценки
            
        Returns:
            Метрики performance
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.act(state, training=False)  # Greedy policy
                next_state, reward, done, _ = env.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        metrics = {
            "eval_reward_mean": np.mean(episode_rewards),
            "eval_reward_std": np.std(episode_rewards),
            "eval_reward_min": np.min(episode_rewards),
            "eval_reward_max": np.max(episode_rewards),
            "eval_length_mean": np.mean(episode_lengths),
            "eval_episodes": num_episodes,
        }
        
        self.logger.info("Evaluation completed", **metrics)
        return metrics
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Получить статистики обучения."""
        return {
            "training_step": self.training_step,
            "epsilon": self.epsilon_schedule.get_epsilon(self.training_step),
            "replay_buffer_size": len(self.replay_buffer),
            "loss_history": self.loss_history[-1000:],  # Последние 1000 значений
            "episode_rewards": self.episode_rewards[-100:],  # Последние 100 эпизодов
            "network_stats": self.q_network.get_network_stats(),
        }
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Сохранение checkpoint агента.
        
        Args:
            filepath: Путь для сохранения
            metadata: Дополнительные метаданные
        """
        checkpoint = {
            "config": self.config.dict(),
            "training_step": self.training_step,
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "episode_rewards": self.episode_rewards,
            "epsilon_schedule_state": self.epsilon_schedule.get_state(),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Сохранение replay buffer отдельно (может быть большим)
        buffer_filepath = filepath.replace(".pth", "_buffer.pkl")
        with open(buffer_filepath, "wb") as f:
            pickle.dump(self.replay_buffer, f)
        
        torch.save(checkpoint, filepath)
        self.logger.info("Checkpoint сохранен", filepath=filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, load_buffer: bool = True) -> 'DQN':
        """
        Загрузка агента из checkpoint.
        
        Args:
            filepath: Путь к checkpoint
            load_buffer: Загружать ли replay buffer
            
        Returns:
            Загруженный агент
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Создание агента с сохраненной конфигурацией
        config = DQNConfig(**checkpoint["config"])
        agent = cls(config)
        
        # Загрузка состояний
        agent.training_step = checkpoint["training_step"]
        agent.q_network.load_state_dict(checkpoint["q_network_state"])
        agent.target_network.load_state_dict(checkpoint["target_network_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.loss_history = checkpoint["loss_history"]
        agent.episode_rewards = checkpoint["episode_rewards"]
        agent.epsilon_schedule.load_state(checkpoint["epsilon_schedule_state"])
        
        # Загрузка replay buffer если нужно
        if load_buffer:
            buffer_filepath = filepath.replace(".pth", "_buffer.pkl")
            if Path(buffer_filepath).exists():
                with open(buffer_filepath, "rb") as f:
                    agent.replay_buffer = pickle.load(f)
        
        agent.logger.info("Checkpoint загружен", filepath=filepath)
        return agent
    
    def to(self, device: Union[str, torch.device]) -> 'DQN':
        """Перемещение агента на другое устройство."""
        self.device = torch.device(device)
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        return self
    
    def __repr__(self) -> str:
        """Строковое представление агента."""
        return (
            f"DQN(state_size={self.config.network_config.state_size}, "
            f"action_size={self.config.network_config.action_size}, "
            f"training_step={self.training_step}, "
            f"device={self.device})"
        )