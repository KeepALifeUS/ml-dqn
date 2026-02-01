"""
Deep Q-Network (DQN) для криптотрейдинга с Context7 enterprise паттернами.

Этот пакет предоставляет полную реализацию DQN и его улучшений:
- Базовый DQN с epsilon-greedy exploration
- Double DQN для устранения overestimation bias
- Dueling DQN с раздельными value и advantage потоками
- Prioritized Experience Replay для эффективного обучения
- Rainbow DQN объединяющий все улучшения
- Специализированная интеграция для криптотрейдинга

Enterprise паттерны Context7:
- Production-ready error handling и logging
- Comprehensive monitoring и metrics
- Scalable architecture с async support
- Type hints и strict validation
- Performance optimization
"""

from typing import Dict, Any
import logging
from pathlib import Path

# Версия пакета
__version__ = "1.0.0"

# Экспорт основных компонентов
from .core.dqn import DQN
from .agents.dqn_trader import DQNTrader
from .training.dqn_trainer import DQNTrainer
from .buffers.replay_buffer import ReplayBuffer
from .buffers.prioritized_replay import PrioritizedReplayBuffer
from .networks.q_network import QNetwork
from .extensions.double_dqn import DoubleDQN
from .extensions.dueling_dqn import DuelingDQN
from .extensions.rainbow_dqn import RainbowDQN

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Создание директории для логов если не существует
PACKAGE_ROOT = Path(__file__).parent.parent
LOG_DIR = PACKAGE_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Экспорт всех публичных компонентов
__all__ = [
    # Core components
    "DQN",
    "DQNTrader", 
    "DQNTrainer",
    
    # Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    
    # Networks
    "QNetwork",
    
    # Extensions
    "DoubleDQN",
    "DuelingDQN", 
    "RainbowDQN",
    
    # Utilities
    "get_package_info",
    "setup_logging",
]


def get_package_info() -> Dict[str, Any]:
    """Получить информацию о пакете."""
    return {
        "name": "ml-dqn",
        "version": __version__,
        "description": "Deep Q-Network implementation for cryptocurrency trading",
        "root_path": str(PACKAGE_ROOT),
        "log_directory": str(LOG_DIR),
    }


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Настроить логирование для пакета.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу логов (опционально)
    """
    import logging.config
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"]
        }
    }
    
    if log_file:
        config["handlers"]["file"] = {
            "level": level,
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": log_file,
            "mode": "a"
        }
        config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(config)


# Инициализация логирования по умолчанию
setup_logging()

# Информация о пакете при импорте
logger = logging.getLogger(__name__)
logger.info(f"Loaded ML-DQN package v{__version__}")
logger.info(f"Package root: {PACKAGE_ROOT}")