# src/data_loader.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и подготовки данных ОСО"""

    def __init__(self, data_path=None):
        if data_path is None:
            # Автоматически определяем путь к файлу данных
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_dir, "..")
            self.data_path = os.path.join(base_dir, "data", "raw", "ОСО_индекс_12.dat")
        else:
            self.data_path = data_path

        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler_x, self.scaler_y = StandardScaler(), StandardScaler()

    def load_data(self):
        """Загрузка данных из файла"""
        try:
            logger.info(f"Загрузка данных из {self.data_path}")
            logger.info(f"Абсолютный путь: {os.path.abspath(self.data_path)}")

            if not os.path.exists(self.data_path):
                logger.error(f"Файл не существует: {self.data_path}")
                return False

            # Чтение файла с запятыми как разделителями
            self.data = pd.read_csv(self.data_path, sep=',', header=None)

            logger.info(f"Данные загружены. Форма: {self.data.shape}")
            logger.info(f"Первые 3 строки данных:")
            print(self.data.head(3))

            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            return False

    def clean_data(self):
        """Очистка и предобработка данных"""
        if self.data is None:
            logger.error("Данные не загружены!")
            return False

        try:
            logger.info("Очистка данных...")

            # Первая строка может содержать заголовки или комментарии
            first_row_str = str(self.data.iloc[0, 0])
            if 'Comments' in first_row_str or 'TO' in first_row_str:
                logger.info("Обнаружена строка с заголовками. Пропускаем первую строку.")
                self.data = self.data.iloc[1:]  # Пропускаем первую строку

            # Если данные в одной колонке, разделим их
            if self.data.shape[1] == 1:
                logger.info("Данные в одной колонке, разделяем...")
                split_data = self.data.iloc[:, 0].str.split(',', expand=True)
                self.data = split_data

            # Преобразуем все данные в числовой формат
            self.data = self.data.apply(pd.to_numeric, errors='coerce')

            # Удаляем строки с NaN значениями
            original_shape = self.data.shape
            self.data = self.data.dropna()
            logger.info(f"Удалено строк с NaN: {original_shape[0] - self.data.shape[0]}")

            logger.info(f"Данные после очистки. Форма: {self.data.shape}")
            logger.info(f"Первые 3 строки после очистки:")
            print(self.data.head(3))

            return True

        except Exception as e:
            logger.error(f"Ошибка очистки данных: {e}")
            return False

    def prepare_data(self, test_size=0.2, random_state=42):
        """Подготовка и разделение данных"""
        if self.data is None:
            logger.error("Данные не загружены!")
            return False

        try:
            # Очищаем данные
            if not self.clean_data():
                return False

            # Проверяем количество столбцов
            logger.info(f"Количество столбцов в данных: {self.data.shape[1]}")
            logger.info(f"Структура данных: {self.data.columns.tolist()}")

            # Определяем какие столбцы использовать
            # Обычно: столбец 0 - индекс, 1 - год, 2-13 - месяцы, 14 - целевая
            if self.data.shape[1] >= 15:
                X = self.data.iloc[:, 2:14].values  # месячные данные
                y = self.data.iloc[:, 14].values.reshape(-1, 1)  # целевая
            elif self.data.shape[1] == 14:
                X = self.data.iloc[:, 1:13].values  # месячные данные
                y = self.data.iloc[:, 13].values.reshape(-1, 1)  # целевая
            elif self.data.shape[1] == 13:
                X = self.data.iloc[:, 0:12].values  # месячные данные
                y = self.data.iloc[:, 12].values.reshape(-1, 1)  # целевая
            else:
                logger.error(f"Неожиданное количество столбцов: {self.data.shape[1]}")
                return False

            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

            # Нормализация данных
            X_scaled = self.scaler_x.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)

            # Разделение на train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y_scaled,
                test_size=test_size,
                random_state=random_state,
                shuffle=False
            )

            logger.info("Данные подготовлены:")
            logger.info(f"  X_train: {self.X_train.shape}")
            logger.info(f"  X_test: {self.X_test.shape}")
            logger.info(f"  y_train: {self.y_train.shape}")
            logger.info(f"  y_test: {self.y_test.shape}")

            return True

        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def inverse_transform_y(self, y_scaled):
        """Обратное преобразование предсказаний в исходный масштаб"""
        return self.scaler_y.inverse_transform(y_scaled)

    def get_original_data(self):
        return self.data