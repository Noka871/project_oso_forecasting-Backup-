# src/logger_config.py
import logging
import os
from logging.handlers import RotatingFileHandler
import sys


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Настройка логгера с выводом в консоль и файл
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Проверяем, не добавлены ли уже обработчики
    if logger.handlers:
        return logger

    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Добавляем обработчик в логгер
    logger.addHandler(console_handler)

    # Обработчик для файла (если указан)
    if log_file:
        # Создаем директорию для логов если не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Ротирующий файловый обработчик (макс 5MB, 5 бэкапов)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


# Создаем главный логгер проекта
def get_main_logger():
    """
    Возвращает основной логгер проекта
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "..", "logs", "project.log")

    return setup_logger('oso_forecasting', log_file)


# Создаем логгер для данных
def get_data_logger():
    """
    Возвращает логгер для операций с данными
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "..", "logs", "data.log")

    return setup_logger('data', log_file)


# Создаем логгер для модели
def get_model_logger():
    """
    Возвращает логгер для операций с моделью
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "..", "logs", "model.log")

    return setup_logger('model', log_file)


# Создаем логгер для обучения
def get_training_logger():
    """
    Возвращает логгер для процесса обучения
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "..", "logs", "training.log")

    return setup_logger('training', log_file)


# Глобальные логгеры
main_logger = get_main_logger()
data_logger = get_data_logger()
model_logger = get_model_logger()
training_logger = get_training_logger()