import logging
import os
from datetime import datetime
import sys
import traceback


class OzoneLogger:
    def __init__(self, name="OzoneForecasting", log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Создаем папку для логов если ее нет
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Обработчик для файла
        log_filename = f"logs/ozone_forecasting_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Обработчик для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Добавляем обработчики если их еще нет
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

    def info(self, message, *args):
        self.logger.info(message, *args)

    def error(self, message, *args):
        self.logger.error(message, *args)

    def warning(self, message, *args):
        self.logger.warning(message, *args)

    def debug(self, message, *args):
        self.logger.debug(message, *args)

    def critical(self, message, *args):
        # Вместо exc_info добавляем traceback вручную
        if isinstance(message, Exception):
            error_msg = f"{str(message)}\n{traceback.format_exc()}"
            self.logger.critical(error_msg)
        else:
            self.logger.critical(message)


# Создаем глобальный экземпляр логгера
logger = OzoneLogger()


def log_function_call(func):
    """Декоратор для логирования вызовов функций"""

    def wrapper(*args, **kwargs):
        logger.info(f"Вызов функции: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Функция {func.__name__} выполнена успешно")
            return result
        except Exception as e:
            error_msg = f"Ошибка в функции {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise

    return wrapper


def log_model_training(model_name):
    """Декоратор для логирования обучения моделей"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"🚀 Начало обучения модели: {model_name}")
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                training_time = datetime.now() - start_time
                logger.info(f"✅ Модель {model_name} обучена успешно. Время: {training_time}")
                return result
            except Exception as e:
                error_msg = f"❌ Ошибка обучения модели {model_name}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise

        return wrapper

    return decorator


def log_data_operation(operation_name):
    """Декоратор для логирования операций с данными"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"📊 Начало операции: {operation_name}")

            try:
                result = func(*args, **kwargs)
                if hasattr(result, 'shape'):
                    logger.info(f"✅ {operation_name} завершена. Размер данных: {result.shape}")
                else:
                    logger.info(f"✅ {operation_name} завершена успешно")
                return result
            except Exception as e:
                error_msg = f"❌ Ошибка операции {operation_name}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise

        return wrapper

    return decorator