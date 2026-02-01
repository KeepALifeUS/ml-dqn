"""
Q-Network архитектура для DQN с Context7 enterprise паттернами.

Реализует стандартную Q-network архитектуру с оптимизациями для криптотрейдинга:
- Dropout для регуляризации
- Batch normalization для стабильности
- Residual connections для глубоких сетей
- Configurable architecture через Pydantic
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator
import numpy as np

logger = logging.getLogger(__name__)


class QNetworkConfig(BaseModel):
    """Конфигурация Q-Network с валидацией."""
    
    state_size: int = Field(..., description="Размерность пространства состояний", gt=0)
    action_size: int = Field(..., description="Размерность пространства действий", gt=0)
    hidden_layers: List[int] = Field(
        default=[512, 256, 128], 
        description="Размеры скрытых слоев"
    )
    dropout_rate: float = Field(default=0.2, description="Вероятность dropout", ge=0.0, le=0.8)
    use_batch_norm: bool = Field(default=True, description="Использовать batch normalization")
    use_residual: bool = Field(default=True, description="Использовать residual connections")
    activation: str = Field(default="relu", description="Функция активации")
    output_activation: Optional[str] = Field(default=None, description="Активация выходного слоя")
    init_type: str = Field(default="xavier_uniform", description="Тип инициализации весов")
    
    @validator("hidden_layers")
    def validate_hidden_layers(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Должен быть минимум один скрытый слой")
        if any(size <= 0 for size in v):
            raise ValueError("Все размеры слоев должны быть положительными")
        return v
    
    @validator("activation")
    def validate_activation(cls, v):
        valid_activations = ["relu", "leaky_relu", "elu", "selu", "gelu", "swish"]
        if v not in valid_activations:
            raise ValueError(f"Активация должна быть одной из: {valid_activations}")
        return v
    
    @validator("init_type")
    def validate_init_type(cls, v):
        valid_inits = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"]
        if v not in valid_inits:
            raise ValueError(f"Тип инициализации должен быть одним из: {valid_inits}")
        return v


class QNetwork(nn.Module):
    """
    Q-Network для DQN с enterprise-grade архитектурой.
    
    Features:
    - Configurable architecture через Pydantic config
    - Dropout regularization для предотвращения overfitting  
    - Batch normalization для стабильности обучения
    - Residual connections для глубоких сетей
    - Proper weight initialization
    - Gradient clipping support
    - Performance monitoring hooks
    """
    
    def __init__(self, config: QNetworkConfig):
        """
        Инициализация Q-Network.
        
        Args:
            config: Конфигурация сети
        """
        super().__init__()
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size
        
        # Построение архитектуры
        self._build_network()
        
        # Инициализация весов
        self._initialize_weights()
        
        # Monitoring hooks для production
        self._register_hooks()
        
        logger.info(f"Создана Q-Network: {self._get_network_info()}")
    
    def _build_network(self) -> None:
        """Построение архитектуры сети."""
        layers = []
        layer_sizes = [self.config.state_size] + self.config.hidden_layers
        
        # Скрытые слои
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(in_size, out_size))
            
            # Batch normalization
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            
            # Activation
            layers.append(self._get_activation())
            
            # Dropout
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Основная сеть
        self.feature_layers = nn.Sequential(*layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(self.config.hidden_layers[-1], self.config.action_size)
        
        # Output activation если нужна
        self.output_activation = None
        if self.config.output_activation:
            self.output_activation = self._get_activation(self.config.output_activation)
        
        # Residual connections если включены
        if self.config.use_residual:
            self._setup_residual_connections()
    
    def _setup_residual_connections(self) -> None:
        """Настройка residual connections."""
        self.residual_layers = nn.ModuleList()
        layer_sizes = [self.config.state_size] + self.config.hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Skip connection только если размеры совпадают
            if in_size == out_size:
                self.residual_layers.append(nn.Identity())
            else:
                # Projection layer для изменения размерности
                self.residual_layers.append(nn.Linear(in_size, out_size))
    
    def _get_activation(self, activation: Optional[str] = None) -> nn.Module:
        """Получить функцию активации."""
        act_name = activation or self.config.activation
        
        activation_map = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "elu": nn.ELU(inplace=True),
            "selu": nn.SELU(inplace=True),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(inplace=True),
        }
        
        return activation_map[act_name]
    
    def _initialize_weights(self) -> None:
        """Инициализация весов сети."""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                if self.config.init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif self.config.init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif self.config.init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity=self.config.activation)
                elif self.config.init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.activation)
                elif self.config.init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                
                # Инициализация bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        self.apply(init_layer)
        
        # Специальная инициализация выходного слоя
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        logger.debug(f"Веса инициализированы методом: {self.config.init_type}")
    
    def _register_hooks(self) -> None:
        """Регистрация hooks для мониторинга."""
        def forward_hook(module, input, output):
            # Проверка на NaN и Inf
            if torch.isnan(output).any():
                logger.error(f"NaN обнаружен в выходе модуля {module.__class__.__name__}")
            if torch.isinf(output).any():
                logger.error(f"Inf обнаружен в выходе модуля {module.__class__.__name__}")
        
        def backward_hook(module, grad_input, grad_output):
            # Проверка градиентов
            if grad_output[0] is not None:
                grad_norm = torch.norm(grad_output[0])
                if grad_norm > 10.0:  # Threshold для gradient explosion
                    logger.warning(f"Большие градиенты в {module.__class__.__name__}: {grad_norm:.4f}")
        
        # Регистрация hooks только в debug режиме
        if logger.isEnabledFor(logging.DEBUG):
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через сеть.
        
        Args:
            state: Tensor состояний [batch_size, state_size]
            
        Returns:
            Q-values для всех действий [batch_size, action_size]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Добавить batch dimension
        
        x = state
        
        # Residual connections если включены
        if self.config.use_residual and hasattr(self, 'residual_layers'):
            residuals = []
            layer_idx = 0
            
            for i, layer in enumerate(self.feature_layers):
                if isinstance(layer, nn.Linear):
                    # Сохранить входные данные для residual connection
                    if layer_idx < len(self.residual_layers):
                        residual = self.residual_layers[layer_idx](x)
                        residuals.append(residual)
                        layer_idx += 1
                    
                    x = layer(x)
                    
                    # Добавить residual connection после активации
                    if len(residuals) > 0 and not isinstance(self.feature_layers[i + 1], nn.BatchNorm1d):
                        x = x + residuals.pop(0)
                
                else:
                    x = layer(x)
        else:
            # Стандартный forward pass
            x = self.feature_layers(x)
        
        # Выходной слой
        q_values = self.output_layer(x)
        
        # Output activation если есть
        if self.output_activation:
            q_values = self.output_activation(q_values)
        
        return q_values
    
    def get_action_values(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Получить Q-values для состояний.
        
        Args:
            state: Состояния [batch_size, state_size]
            action: Действия (если None, возвращает все Q-values) [batch_size]
            
        Returns:
            Q-values [batch_size] или [batch_size, action_size]
        """
        q_values = self.forward(state)
        
        if action is not None:
            # Выбрать Q-values для конкретных действий
            if action.dim() == 1:
                q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            else:
                q_values = q_values.gather(1, action)
        
        return q_values
    
    def get_best_actions(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получить лучшие действия и их Q-values.
        
        Args:
            state: Состояния [batch_size, state_size]
            
        Returns:
            Tuple из (actions, q_values)
        """
        q_values = self.forward(state)
        actions = torch.argmax(q_values, dim=1)
        max_q_values = torch.max(q_values, dim=1)[0]
        
        return actions, max_q_values
    
    def soft_update(self, target_network: 'QNetwork', tau: float = 0.005) -> None:
        """
        Мягкое обновление целевой сети.
        
        Args:
            target_network: Целевая сеть для обновления
            tau: Коэффициент интерполяции
        """
        for target_param, local_param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, target_network: 'QNetwork') -> None:
        """
        Жесткое обновление целевой сети.
        
        Args:
            target_network: Целевая сеть для обновления
        """
        target_network.load_state_dict(self.state_dict())
    
    def clip_gradients(self, max_norm: float = 1.0) -> float:
        """
        Обрезание градиентов для стабильности обучения.
        
        Args:
            max_norm: Максимальная норма градиентов
            
        Returns:
            Норма градиентов до обрезания
        """
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Заморозка определенных слоев.
        
        Args:
            layer_names: Список имен слоев для заморозки
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Заморожен слой: {name}")
    
    def unfreeze_all(self) -> None:
        """Разморозка всех слоев."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Все слои разморожены")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Получить статистику сети.
        
        Returns:
            Словарь со статистикой
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "memory_mb": total_params * 4 / (1024 * 1024),  # Примерно для float32
            "config": self.config.dict(),
        }
    
    def _get_network_info(self) -> str:
        """Получить краткую информацию о сети."""
        stats = self.get_network_stats()
        return (
            f"State:{self.state_size}, Action:{self.action_size}, "
            f"Layers:{self.config.hidden_layers}, "
            f"Params:{stats['total_parameters']:,}, "
            f"Memory:{stats['memory_mb']:.1f}MB"
        )
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Сохранение checkpoint модели.
        
        Args:
            filepath: Путь для сохранения
            metadata: Дополнительные метаданные
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.dict(),
            "network_stats": self.get_network_stats(),
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint сохранен: {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> Tuple['QNetwork', Dict]:
        """
        Загрузка модели из checkpoint.
        
        Args:
            filepath: Путь к checkpoint
            
        Returns:
            Tuple из (model, metadata)
        """
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Создание модели с сохраненной конфигурацией
        config = QNetworkConfig(**checkpoint["config"])
        model = cls(config)
        
        # Загрузка весов
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Checkpoint загружен: {filepath}")
        
        return model, checkpoint.get("metadata", {})