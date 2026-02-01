"""
Dueling Network Architecture для DQN с Context7 enterprise паттернами.

Dueling DQN разделяет Q-function на value и advantage потоки:
- Value stream оценивает "goodness" состояния
- Advantage stream оценивает relative advantage действий  
- Лучшая generalization через explicit value decomposition
- Improved learning efficiency для states с similar action values
"""

import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator
import numpy as np
import structlog

from .q_network import QNetworkConfig

logger = structlog.get_logger(__name__)


class DuelingNetworkConfig(QNetworkConfig):
    """Конфигурация Dueling Network с дополнительными параметрами."""
    
    # Dueling-specific parameters
    value_hidden_layers: Optional[list] = Field(
        default=None, 
        description="Hidden layers для value stream (если None, используется половина от общих)"
    )
    advantage_hidden_layers: Optional[list] = Field(
        default=None,
        description="Hidden layers для advantage stream (если None, используется половина от общих)"
    )
    
    # Aggregation method
    aggregation_method: str = Field(
        default="mean", 
        description="Метод агрегации value и advantage"
    )
    
    # Stream separation point
    separation_layer: int = Field(
        default=-2, 
        description="Слой где разделяются потоки (отрицательный от конца)"
    )
    
    # Normalization options
    normalize_advantage: bool = Field(
        default=True, 
        description="Нормализация advantage stream"
    )
    
    @validator("aggregation_method")
    def validate_aggregation(cls, v):
        valid_methods = ["mean", "max", "naive"]
        if v not in valid_methods:
            raise ValueError(f"Aggregation method должен быть одним из: {valid_methods}")
        return v


class DuelingNetwork(nn.Module):
    """
    Dueling Network архитектура с enterprise-grade реализацией.
    
    Architecture:
    - Shared feature layers для общего representation learning
    - Value stream для оценки state value V(s) 
    - Advantage stream для оценки action advantages A(s,a)
    - Aggregation layer: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    
    Features:
    - Configurable stream architectures
    - Multiple aggregation methods (mean, max, naive)
    - Proper initialization для stable training
    - Comprehensive monitoring hooks
    - Memory-efficient implementation
    """
    
    def __init__(self, config: DuelingNetworkConfig):
        """
        Инициализация Dueling Network.
        
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
        
        # Monitoring hooks
        self._register_hooks()
        
        self.logger = structlog.get_logger(__name__).bind(
            component="DuelingNetwork",
            aggregation=config.aggregation_method
        )
        
        self.logger.info(f"Создана Dueling Network: {self._get_network_info()}")
    
    def _build_network(self) -> None:
        """Построение dueling архитектуры."""
        # Определение точки разделения
        layer_sizes = [self.config.state_size] + self.config.hidden_layers
        separation_idx = len(layer_sizes) + self.config.separation_layer
        separation_idx = max(0, min(separation_idx, len(layer_sizes) - 1))
        
        # Shared feature layers
        self.shared_layers = nn.ModuleList()
        for i in range(separation_idx):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            layer_block = nn.Sequential()
            layer_block.add_module("linear", nn.Linear(in_size, out_size))
            
            if self.config.use_batch_norm:
                layer_block.add_module("batch_norm", nn.BatchNorm1d(out_size))
            
            layer_block.add_module("activation", self._get_activation())
            
            if self.config.dropout_rate > 0:
                layer_block.add_module("dropout", nn.Dropout(self.config.dropout_rate))
            
            self.shared_layers.append(layer_block)
        
        # Feature size после shared layers
        shared_feature_size = layer_sizes[separation_idx + 1]
        
        # Value stream
        value_layers = (self.config.value_hidden_layers or 
                       self.config.hidden_layers[separation_idx + 1:] or 
                       [shared_feature_size // 2])
        
        self.value_stream = self._build_stream(
            shared_feature_size, value_layers, output_size=1, name="value"
        )
        
        # Advantage stream  
        advantage_layers = (self.config.advantage_hidden_layers or
                           self.config.hidden_layers[separation_idx + 1:] or
                           [shared_feature_size // 2])
        
        self.advantage_stream = self._build_stream(
            shared_feature_size, advantage_layers, 
            output_size=self.action_size, name="advantage"
        )
    
    def _build_stream(self, input_size: int, hidden_layers: list, 
                     output_size: int, name: str) -> nn.Sequential:
        """
        Построение stream (value или advantage).
        
        Args:
            input_size: Размер входа
            hidden_layers: Размеры скрытых слоев
            output_size: Размер выхода
            name: Название stream
            
        Returns:
            Sequential model для stream
        """
        layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            is_output_layer = (i == len(layer_sizes) - 2)
            
            # Linear layer
            layers.append(nn.Linear(in_size, out_size))
            
            # Batch normalization (не на выходном слое)
            if not is_output_layer and self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            
            # Activation (не на выходном слое или с output activation)
            if not is_output_layer:
                layers.append(self._get_activation())
                
                # Dropout
                if self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
            elif self.config.output_activation:
                layers.append(self._get_activation(self.config.output_activation))
        
        stream = nn.Sequential(*layers)
        
        self.logger.debug(f"{name.capitalize()} stream created", 
                         layers=len(layer_sizes) - 1,
                         params=sum(p.numel() for p in stream.parameters()))
        
        return stream
    
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
        """Специализированная инициализация весов для dueling architecture."""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                # Shared layers - standard initialization
                if any(m is layer for layer_block in self.shared_layers for layer in layer_block):
                    if self.config.init_type == "xavier_uniform":
                        nn.init.xavier_uniform_(m.weight)
                    elif self.config.init_type == "kaiming_uniform":
                        nn.init.kaiming_uniform_(m.weight, nonlinearity=self.config.activation)
                
                # Value stream - smaller initialization (stable baseline)
                elif any(m is layer for layer in self.value_stream):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                
                # Advantage stream - zero initialization для последнего слоя
                elif any(m is layer for layer in self.advantage_stream):
                    if m is list(self.advantage_stream.modules())[-1]:  # Last layer
                        nn.init.zeros_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                
                # Bias initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        self.apply(init_layer)
        self.logger.debug(f"Dueling network weights initialized")
    
    def _register_hooks(self) -> None:
        """Регистрация monitoring hooks."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        
        def stream_hook(name):
            def hook_fn(module, input, output):
                if torch.isnan(output).any():
                    self.logger.error(f"NaN в {name} stream")
                if torch.isinf(output).any():
                    self.logger.error(f"Inf в {name} stream")
                
                # Statistics
                self.logger.debug(f"{name} stream stats",
                                mean=output.mean().item(),
                                std=output.std().item(),
                                min=output.min().item(),
                                max=output.max().item())
            return hook_fn
        
        # Hooks для streams
        self.value_stream.register_forward_hook(stream_hook("value"))
        self.advantage_stream.register_forward_hook(stream_hook("advantage"))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через dueling architecture.
        
        Args:
            state: Tensor состояний [batch_size, state_size]
            
        Returns:
            Q-values [batch_size, action_size]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Shared feature extraction
        x = state
        for layer_block in self.shared_layers:
            x = layer_block(x)
        
        # Stream computation
        value = self.value_stream(x)  # [batch_size, 1]
        advantage = self.advantage_stream(x)  # [batch_size, action_size]
        
        # Advantage normalization
        if self.config.normalize_advantage:
            if self.config.aggregation_method == "mean":
                advantage_normalized = advantage - advantage.mean(dim=1, keepdim=True)
            elif self.config.aggregation_method == "max":
                advantage_normalized = advantage - advantage.max(dim=1, keepdim=True)[0].unsqueeze(1)
            else:  # naive
                advantage_normalized = advantage
        else:
            advantage_normalized = advantage
        
        # Q-value computation: Q(s,a) = V(s) + A(s,a)
        q_values = value + advantage_normalized
        
        return q_values
    
    def forward_with_streams(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass с возвращением отдельных streams для анализа.
        
        Args:
            state: Tensor состояний
            
        Returns:
            Tuple из (q_values, value, advantage)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Shared features
        x = state
        for layer_block in self.shared_layers:
            x = layer_block(x)
        
        # Streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q-values
        if self.config.normalize_advantage:
            if self.config.aggregation_method == "mean":
                advantage_norm = advantage - advantage.mean(dim=1, keepdim=True)
            elif self.config.aggregation_method == "max":
                advantage_norm = advantage - advantage.max(dim=1, keepdim=True)[0].unsqueeze(1)
            else:
                advantage_norm = advantage
        else:
            advantage_norm = advantage
        
        q_values = value + advantage_norm
        
        return q_values, value, advantage
    
    def get_stream_statistics(self, state_batch: torch.Tensor) -> Dict[str, Any]:
        """
        Анализ streams для monitoring и debugging.
        
        Args:
            state_batch: Batch состояний для анализа
            
        Returns:
            Статистики streams
        """
        with torch.no_grad():
            q_values, value, advantage = self.forward_with_streams(state_batch)
            
            stats = {
                "value_mean": value.mean().item(),
                "value_std": value.std().item(),
                "value_min": value.min().item(),
                "value_max": value.max().item(),
                
                "advantage_mean": advantage.mean().item(),
                "advantage_std": advantage.std().item(),  
                "advantage_min": advantage.min().item(),
                "advantage_max": advantage.max().item(),
                
                "q_values_mean": q_values.mean().item(),
                "q_values_std": q_values.std().item(),
                
                # Stream correlation
                "value_advantage_corr": torch.corrcoef(torch.stack([
                    value.flatten(), advantage.mean(dim=1)
                ]))[0, 1].item(),
                
                # Advantage распределение
                "advantage_entropy": -(F.softmax(advantage, dim=1) * 
                                     F.log_softmax(advantage, dim=1)).sum(dim=1).mean().item(),
            }
        
        return stats
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Статистики сети с dueling-специфичной информацией."""
        total_params = sum(p.numel() for p in self.parameters())
        shared_params = sum(p.numel() for layer_block in self.shared_layers for p in layer_block.parameters())
        value_params = sum(p.numel() for p in self.value_stream.parameters())
        advantage_params = sum(p.numel() for p in self.advantage_stream.parameters())
        
        return {
            "total_parameters": total_params,
            "shared_parameters": shared_params,
            "value_parameters": value_params,
            "advantage_parameters": advantage_params,
            
            "parameter_distribution": {
                "shared_ratio": shared_params / total_params,
                "value_ratio": value_params / total_params,  
                "advantage_ratio": advantage_params / total_params,
            },
            
            "architecture": {
                "shared_layers": len(self.shared_layers),
                "value_layers": len(self.value_stream),
                "advantage_layers": len(self.advantage_stream),
                "separation_point": self.config.separation_layer,
                "aggregation_method": self.config.aggregation_method,
            },
            
            "memory_mb": total_params * 4 / (1024 * 1024),
            "config": self.config.dict(),
        }
    
    def _get_network_info(self) -> str:
        """Краткая информация о dueling network."""
        stats = self.get_network_stats()
        return (
            f"DuelingNetwork(State:{self.state_size}, Action:{self.action_size}, "
            f"Shared:{stats['shared_parameters']:,}, "
            f"Value:{stats['value_parameters']:,}, "
            f"Advantage:{stats['advantage_parameters']:,}, "
            f"Aggregation:{self.config.aggregation_method})"
        )
    
    def __repr__(self) -> str:
        """Строковое представление dueling network."""
        return self._get_network_info()