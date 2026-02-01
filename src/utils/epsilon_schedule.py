"""
Epsilon Scheduling для DQN exploration с Context7 enterprise паттернами.

Реализует различные стратегии epsilon decay для оптимальной exploration:
- Linear decay с configurable schedule
- Exponential decay с adaptive rates  
- Step-wise decay с milestone-based reductions
- Cosine annealing с warm restarts
- Custom schedules через lambda functions
- Production monitoring и logging
"""

import logging
from typing import Optional, Callable, Dict, Any, List
import numpy as np
import math
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ScheduleType(str, Enum):
    """Типы epsilon schedules."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"
    POLYNOMIAL = "polynomial"
    CUSTOM = "custom"


@dataclass
class ScheduleState:
    """Состояние epsilon schedule для persistence."""
    current_step: int
    current_epsilon: float
    schedule_type: str
    parameters: Dict[str, Any]


class EpsilonScheduleConfig(BaseModel):
    """Конфигурация epsilon schedule с валидацией."""
    
    # Basic parameters
    start_epsilon: float = Field(default=1.0, description="Начальный epsilon", ge=0, le=1.0)
    end_epsilon: float = Field(default=0.01, description="Конечный epsilon", ge=0, le=1.0)
    
    # Schedule type и parameters
    schedule_type: ScheduleType = Field(default=ScheduleType.EXPONENTIAL, description="Тип schedule")
    
    # Linear schedule
    total_steps: Optional[int] = Field(default=100000, description="Общее количество шагов", gt=0)
    
    # Exponential schedule  
    decay_rate: float = Field(default=0.995, description="Exponential decay rate", ge=0.9, le=1.0)
    
    # Step schedule
    step_sizes: Optional[List[int]] = Field(default=None, description="Sizes для step decay")
    step_gammas: Optional[List[float]] = Field(default=None, description="Multipliers для step decay")
    
    # Cosine schedule
    cosine_restarts: bool = Field(default=False, description="Warm restarts для cosine")
    restart_periods: Optional[List[int]] = Field(default=None, description="Periods для restarts")
    
    # Polynomial schedule
    power: float = Field(default=1.0, description="Power для polynomial decay", gt=0)
    
    # General parameters
    min_epsilon: float = Field(default=0.001, description="Абсолютный минимум epsilon", ge=0, le=0.1)
    warmup_steps: int = Field(default=0, description="Steps для warmup", ge=0)
    
    @validator("end_epsilon")
    def validate_end_epsilon(cls, v, values):
        if "start_epsilon" in values and v >= values["start_epsilon"]:
            raise ValueError("end_epsilon должен быть < start_epsilon")
        return v
    
    @validator("min_epsilon")
    def validate_min_epsilon(cls, v, values):
        if "end_epsilon" in values and v > values["end_epsilon"]:
            raise ValueError("min_epsilon должен быть <= end_epsilon")
        return v
    
    @validator("step_gammas")
    def validate_step_gammas(cls, v, values):
        if v is not None and "step_sizes" in values:
            if values["step_sizes"] is not None and len(v) != len(values["step_sizes"]):
                raise ValueError("step_gammas должен иметь такую же длину как step_sizes")
        return v


class BaseEpsilonSchedule(ABC):
    """Базовый класс для epsilon schedules."""
    
    def __init__(self, config: EpsilonScheduleConfig):
        self.config = config
        self.current_step = 0
        self.logger = structlog.get_logger(__name__).bind(
            component="EpsilonSchedule",
            schedule_type=config.schedule_type
        )
    
    @abstractmethod
    def get_epsilon(self, step: Optional[int] = None) -> float:
        """Получить epsilon для заданного step."""
        pass
    
    def step(self) -> float:
        """Increment step и получить новый epsilon."""
        self.current_step += 1
        return self.get_epsilon()
    
    def reset(self) -> None:
        """Сброс schedule к начальному состоянию."""
        self.current_step = 0
    
    def get_state(self) -> ScheduleState:
        """Получить текущее состояние для persistence."""
        return ScheduleState(
            current_step=self.current_step,
            current_epsilon=self.get_epsilon(),
            schedule_type=self.config.schedule_type.value,
            parameters=self.config.dict()
        )
    
    def load_state(self, state: ScheduleState) -> None:
        """Загрузить состояние из persistence."""
        self.current_step = state.current_step


class LinearSchedule(BaseEpsilonSchedule):
    """Linear epsilon decay."""
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        current = step if step is not None else self.current_step
        
        # Warmup phase
        if current < self.config.warmup_steps:
            return self.config.start_epsilon
        
        effective_step = current - self.config.warmup_steps
        total_steps = self.config.total_steps - self.config.warmup_steps
        
        # Linear interpolation
        progress = min(effective_step / total_steps, 1.0)
        epsilon = self.config.start_epsilon - progress * (
            self.config.start_epsilon - self.config.end_epsilon
        )
        
        return max(epsilon, self.config.min_epsilon)


class ExponentialSchedule(BaseEpsilonSchedule):
    """Exponential epsilon decay."""
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        current = step if step is not None else self.current_step
        
        # Warmup phase
        if current < self.config.warmup_steps:
            return self.config.start_epsilon
        
        effective_step = current - self.config.warmup_steps
        
        # Exponential decay
        epsilon = self.config.end_epsilon + (
            self.config.start_epsilon - self.config.end_epsilon
        ) * (self.config.decay_rate ** effective_step)
        
        return max(epsilon, self.config.min_epsilon)


class CosineSchedule(BaseEpsilonSchedule):
    """Cosine annealing epsilon schedule с optional restarts."""
    
    def __init__(self, config: EpsilonScheduleConfig):
        super().__init__(config)
        self.restart_step = 0
        self.restart_count = 0
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        current = step if step is not None else self.current_step
        
        # Warmup phase
        if current < self.config.warmup_steps:
            return self.config.start_epsilon
        
        effective_step = current - self.config.warmup_steps
        
        if self.config.cosine_restarts and self.config.restart_periods:
            # Определение текущего restart period
            period = self.config.restart_periods[
                min(self.restart_count, len(self.config.restart_periods) - 1)
            ]
            
            if effective_step >= self.restart_step + period:
                self.restart_step = effective_step
                self.restart_count += 1
            
            step_in_period = effective_step - self.restart_step
            progress = step_in_period / period
        else:
            # Standard cosine schedule
            total_steps = self.config.total_steps - self.config.warmup_steps
            progress = min(effective_step / total_steps, 1.0)
        
        # Cosine annealing
        epsilon = self.config.end_epsilon + 0.5 * (
            self.config.start_epsilon - self.config.end_epsilon
        ) * (1 + math.cos(math.pi * progress))
        
        return max(epsilon, self.config.min_epsilon)


class StepSchedule(BaseEpsilonSchedule):
    """Step-wise epsilon decay at milestones."""
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        current = step if step is not None else self.current_step
        
        # Warmup phase
        if current < self.config.warmup_steps:
            return self.config.start_epsilon
        
        effective_step = current - self.config.warmup_steps
        
        # Определение текущего step multiplier
        epsilon = self.config.start_epsilon
        
        if self.config.step_sizes and self.config.step_gammas:
            cumulative_step = 0
            for step_size, gamma in zip(self.config.step_sizes, self.config.step_gammas):
                cumulative_step += step_size
                if effective_step >= cumulative_step:
                    epsilon *= gamma
                else:
                    break
        
        # Ensure не ниже end_epsilon
        epsilon = max(epsilon, self.config.end_epsilon)
        return max(epsilon, self.config.min_epsilon)


class PolynomialSchedule(BaseEpsilonSchedule):
    """Polynomial epsilon decay."""
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        current = step if step is not None else self.current_step
        
        # Warmup phase
        if current < self.config.warmup_steps:
            return self.config.start_epsilon
        
        effective_step = current - self.config.warmup_steps
        total_steps = self.config.total_steps - self.config.warmup_steps
        
        # Polynomial decay
        progress = min(effective_step / total_steps, 1.0)
        epsilon = self.config.end_epsilon + (
            self.config.start_epsilon - self.config.end_epsilon
        ) * ((1 - progress) ** self.config.power)
        
        return max(epsilon, self.config.min_epsilon)


class CustomSchedule(BaseEpsilonSchedule):
    """Custom epsilon schedule через lambda function."""
    
    def __init__(self, config: EpsilonScheduleConfig, schedule_fn: Callable[[int], float]):
        super().__init__(config)
        self.schedule_fn = schedule_fn
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        current = step if step is not None else self.current_step
        
        # Warmup phase
        if current < self.config.warmup_steps:
            return self.config.start_epsilon
        
        effective_step = current - self.config.warmup_steps
        epsilon = self.schedule_fn(effective_step)
        
        return max(epsilon, self.config.min_epsilon)


class EpsilonSchedule:
    """
    Factory class для создания epsilon schedules с enterprise функциональностью.
    
    Features:
    - Multiple schedule types (linear, exponential, cosine, step, polynomial)
    - Configurable parameters через Pydantic validation
    - Warmup periods для stable start
    - Hard минимальные limits для safety
    - Comprehensive logging и monitoring
    - State persistence для checkpointing
    - Custom schedules через lambda functions
    """
    
    def __init__(self, 
                 config: Optional[EpsilonScheduleConfig] = None,
                 start_epsilon: float = 1.0,
                 end_epsilon: float = 0.01,
                 decay_rate: float = 0.995,
                 schedule_type: ScheduleType = ScheduleType.EXPONENTIAL,
                 custom_schedule: Optional[Callable[[int], float]] = None):
        """
        Инициализация epsilon schedule.
        
        Args:
            config: Полная конфигурация (приоритетнее individual parameters)
            start_epsilon: Начальный epsilon (backward compatibility)
            end_epsilon: Конечный epsilon (backward compatibility) 
            decay_rate: Decay rate (backward compatibility)
            schedule_type: Тип schedule
            custom_schedule: Custom schedule function
        """
        # Backward compatibility
        if config is None:
            config = EpsilonScheduleConfig(
                start_epsilon=start_epsilon,
                end_epsilon=end_epsilon,
                decay_rate=decay_rate,
                schedule_type=schedule_type
            )
        
        self.config = config
        self.current_step = 0
        
        # Создание соответствующего schedule
        self.schedule = self._create_schedule(custom_schedule)
        
        self.logger = structlog.get_logger(__name__).bind(
            component="EpsilonScheduleFactory",
            schedule_type=config.schedule_type.value
        )
        
        self.logger.info("Epsilon schedule создан", config=config.dict())
    
    def _create_schedule(self, custom_fn: Optional[Callable]) -> BaseEpsilonSchedule:
        """Создание конкретного schedule на основе конфигурации."""
        if self.config.schedule_type == ScheduleType.LINEAR:
            return LinearSchedule(self.config)
        elif self.config.schedule_type == ScheduleType.EXPONENTIAL:
            return ExponentialSchedule(self.config)
        elif self.config.schedule_type == ScheduleType.COSINE:
            return CosineSchedule(self.config)
        elif self.config.schedule_type == ScheduleType.STEP:
            return StepSchedule(self.config)
        elif self.config.schedule_type == ScheduleType.POLYNOMIAL:
            return PolynomialSchedule(self.config)
        elif self.config.schedule_type == ScheduleType.CUSTOM:
            if custom_fn is None:
                raise ValueError("Custom schedule требует schedule function")
            return CustomSchedule(self.config, custom_fn)
        else:
            raise ValueError(f"Неизвестный schedule type: {self.config.schedule_type}")
    
    def get_epsilon(self, step: Optional[int] = None) -> float:
        """
        Получить epsilon для заданного step.
        
        Args:
            step: Текущий step (если None, используется internal counter)
            
        Returns:
            Epsilon value
        """
        if step is not None:
            self.current_step = step
        
        epsilon = self.schedule.get_epsilon(self.current_step)
        
        # Logging для monitoring (реже для performance)
        if self.current_step % 10000 == 0:
            self.logger.debug("Epsilon update", 
                            step=self.current_step, 
                            epsilon=epsilon)
        
        return epsilon
    
    def step(self) -> float:
        """Increment step и получить epsilon."""
        self.current_step += 1
        return self.get_epsilon()
    
    def reset(self) -> None:
        """Сброс schedule к начальному состоянию."""
        self.current_step = 0
        self.schedule.reset()
        self.logger.info("Epsilon schedule сброшен")
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Получить информацию о schedule."""
        return {
            "schedule_type": self.config.schedule_type.value,
            "current_step": self.current_step,
            "current_epsilon": self.get_epsilon(),
            "start_epsilon": self.config.start_epsilon,
            "end_epsilon": self.config.end_epsilon,
            "min_epsilon": self.config.min_epsilon,
            "config": self.config.dict(),
        }
    
    def get_epsilon_trajectory(self, max_steps: int, step_size: int = 1000) -> Dict[str, List]:
        """
        Получить траекторию epsilon для visualization.
        
        Args:
            max_steps: Максимальное количество steps
            step_size: Шаг для sampling
            
        Returns:
            Словарь с steps и epsilon values
        """
        steps = list(range(0, max_steps + 1, step_size))
        epsilons = [self.schedule.get_epsilon(step) for step in steps]
        
        return {"steps": steps, "epsilons": epsilons}
    
    def get_state(self) -> ScheduleState:
        """Получить состояние для persistence."""
        return self.schedule.get_state()
    
    def load_state(self, state: ScheduleState) -> None:
        """Загрузить состояние из persistence."""
        self.current_step = state.current_step
        self.schedule.load_state(state)
        self.logger.info("Состояние epsilon schedule загружено", step=self.current_step)
    
    @classmethod
    def create_linear(cls, start_epsilon: float = 1.0, end_epsilon: float = 0.01, 
                     total_steps: int = 100000) -> 'EpsilonSchedule':
        """Создать linear schedule."""
        config = EpsilonScheduleConfig(
            start_epsilon=start_epsilon,
            end_epsilon=end_epsilon,
            total_steps=total_steps,
            schedule_type=ScheduleType.LINEAR
        )
        return cls(config=config)
    
    @classmethod
    def create_exponential(cls, start_epsilon: float = 1.0, end_epsilon: float = 0.01,
                          decay_rate: float = 0.995) -> 'EpsilonSchedule':
        """Создать exponential schedule."""
        config = EpsilonScheduleConfig(
            start_epsilon=start_epsilon,
            end_epsilon=end_epsilon,
            decay_rate=decay_rate,
            schedule_type=ScheduleType.EXPONENTIAL
        )
        return cls(config=config)
    
    @classmethod 
    def create_cosine(cls, start_epsilon: float = 1.0, end_epsilon: float = 0.01,
                     total_steps: int = 100000, restarts: bool = False) -> 'EpsilonSchedule':
        """Создать cosine schedule."""
        config = EpsilonScheduleConfig(
            start_epsilon=start_epsilon,
            end_epsilon=end_epsilon,
            total_steps=total_steps,
            schedule_type=ScheduleType.COSINE,
            cosine_restarts=restarts
        )
        return cls(config=config)
    
    def __repr__(self) -> str:
        """Строковое представление schedule."""
        return (
            f"EpsilonSchedule(type={self.config.schedule_type.value}, "
            f"step={self.current_step}, epsilon={self.get_epsilon():.4f})"
        )