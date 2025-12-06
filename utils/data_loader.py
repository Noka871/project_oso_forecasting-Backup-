import pandas as pd
import numpy as np
import os
from utils.logger import log_function_call, log_data_operation, logger


class OzoneDataLoader:
    def __init__(self):
        self.data_path = "data/ram/"
        logger.info("Инициализирован OzoneDataLoader")

    @log_data_operation("Создание демонстрационных данных ОСО")
    def create_demo_oso_data(self):
        """Создание демонстрационных данных ОСО"""
        years = range(1960, 2025)
        months = range(1, 13)

        logger.info(f"Генерация данных за период: {years[0]}-{years[-1]}")

        data = []
        for year in years:
            for month in months:
                base_oso = 300
                seasonal = 20 * np.sin((month - 3) * 2 * np.pi / 12)
                trend = -0.1 * (year - 1960)
                noise = np.random.normal(0, 5)

                if year >= 2020 and month in [9, 10, 11, 12]:
                    anomaly = -8
                    logger.debug(f"Аномалия для {year}-{month}: {anomaly}")
                else:
                    anomaly = 0

                oso_value = base_oso + seasonal + trend + noise + anomaly

                data.append({
                    'year': year,
                    'month': month,
                    'oso': max(250, min(350, oso_value)),
                    'latitude': 56.5,
                    'longitude': 84.95,
                    'temperature': 15 + 20 * np.sin((month - 1) * 2 * np.pi / 12) + np.random.normal(0, 3),
                    'pressure': 1013 + np.random.normal(0, 8)
                })

        df = pd.DataFrame(data)
        logger.info(f"Создано демонстрационных данных: {len(df)} записей")
        return df

    @log_data_operation("Создание демонстрационных индексных данных")
    def create_demo_index_data(self):
        """Создание демонстрационных индексных данных"""
        years = range(1960, 2025)

        data = []
        for year in years:
            for month in range(1, 13):
                data.append({
                    'year': year,
                    'month': month,
                    'oso_index': (month + year % 3) % 12 + 1,
                    'seasonal_factor': 0.8 + 0.3 * np.sin((month - 1) * 2 * np.pi / 12),
                    'trend_component': (year - 1960) * 0.02,
                    'anomaly_flag': 1 if (year >= 2020 and month in [9, 10, 11, 12]) else 0
                })

        df = pd.DataFrame(data)
        logger.info(f"Создано индексных данных: {len(df)} записей")
        return df

    @log_data_operation("Загрузка файла ОСО_predict.dat")
    def load_oso_predict(self, file_path=None):
        """Загрузка файла ОСО_predict.dat"""
        if file_path is None:
            file_path = os.path.join(self.data_path, "ОСО_predict.dat")

        logger.info(f"Попытка загрузки файла: {file_path}")

        if os.path.exists(file_path):
            try:
                # Пробуем разные форматы
                for delimiter in ['\s+', '\t', ',', ';']:
                    try:
                        data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                        if len(data.columns) > 1:
                            logger.info(f"Файл успешно загружен с разделителем: {delimiter}")
                            logger.info(f"Структура данных: {data.shape}, колонки: {list(data.columns)}")
                            return data
                    except Exception as e:
                        logger.debug(f"Не удалось загрузить с разделителем {delimiter}: {e}")
                        continue

                # Если не получилось, пробуем фиксированную ширину
                data = pd.read_fwf(file_path, encoding='utf-8')
                logger.info("Файл загружен с фиксированной шириной колонок")
                return data

            except Exception as e:
                logger.error(f"Критическая ошибка загрузки файла: {e}")
                logger.warning("Создание демонстрационных данных вместо загрузки")
                return self.create_demo_oso_data()
        else:
            logger.warning(f"Файл не найден: {file_path}")
            logger.info("Создание демонстрационных данных")
            return self.create_demo_oso_data()

    @log_data_operation("Загрузка файла ОСО_индекс_12.dat")
    def load_oso_index_12(self, file_path=None):
        """Загрузка файла ОСО_индекс_12.dat"""
        if file_path is None:
            file_path = os.path.join(self.data_path, "ОСО_индекс_12.dat")

        logger.info(f"Попытка загрузки индексного файла: {file_path}")

        if os.path.exists(file_path):
            try:
                for delimiter in ['\s+', '\t', ',', ';']:
                    try:
                        data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                        if len(data.columns) > 1:
                            logger.info(f"Индексный файл загружен с разделителем: {delimiter}")
                            return data
                    except:
                        continue

                data = pd.read_fwf(file_path, encoding='utf-8')
                logger.info("Индексный файл загружен с фиксированной шириной колонок")
                return data

            except Exception as e:
                logger.error(f"Ошибка загрузки индексного файла: {e}")
                logger.warning("Создание демонстрационных индексных данных")
                return self.create_demo_index_data()
        else:
            logger.warning(f"Индексный файл не найден: {file_path}")
            logger.info("Создание демонстрационных индексных данных")
            return self.create_demo_index_data()

    @log_function_call
    def analyze_data(self, data):
        """Анализ данных"""
        logger.info(f"Анализ данных: {len(data)} записей")

        analysis = {
            'total_records': len(data),
            'columns': list(data.columns),
            'date_range': None,
            'oso_stats': None
        }

        if 'year' in data.columns and 'month' in data.columns:
            years = data['year'].unique()
            analysis['date_range'] = f"{min(years)}-{max(years)}"
            logger.info(f"Период данных: {analysis['date_range']}")

        if 'oso' in data.columns:
            analysis['oso_stats'] = {
                'mean': data['oso'].mean(),
                'min': data['oso'].min(),
                'max': data['oso'].max(),
                'std': data['oso'].std()
            }
            logger.info(f"Статистика ОСО: среднее={analysis['oso_stats']['mean']:.1f}, "
                        f"min={analysis['oso_stats']['min']:.1f}, max={analysis['oso_stats']['max']:.1f}")

        return analysis