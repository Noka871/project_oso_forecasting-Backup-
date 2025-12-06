"""
Сравнительный анализ различных архитектур нейронных сетей
для прогнозирования временных рядов (данные озонового слоя)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, Bidirectional
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import json
import os

from utils.data_loader import OzoneDataLoader
from utils.logger import logger


class ModelComparator:
    def __init__(self):
        self.data_loader = OzoneDataLoader()
        self.models = {}
        self.results = {}
        logger.info("Инициализирован ModelComparator для сравнения архитектур")

    def prepare_data(self, sequence_length=12):
        """Подготовка данных для всех экспериментов"""
        data = self.data_loader.create_demo_oso_data()
        values = data['oso'].values

        X, y = [], []
        for i in range(len(values) - sequence_length):
            X.append(values[i:(i + sequence_length)])
            y.append(values[i + sequence_length])

        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Разделение на train/validation
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Данные подготовлены: train={self.X_train.shape}, val={self.X_val.shape}")
        return self.X_train, self.y_train, self.X_val, self.y_val

    def build_models(self):
        """Создание различных архитектур для сравнения"""

        # 1. Простая LSTM
        lstm_model = Sequential([
            LSTM(64, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['LSTM'] = lstm_model

        # 2. Глубокая LSTM
        deep_lstm = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        deep_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['Deep_LSTM'] = deep_lstm

        # 3. Bidirectional LSTM
        bi_lstm = Sequential([
            Bidirectional(LSTM(64), input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        bi_lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['Bidirectional_LSTM'] = bi_lstm

        # 4. GRU
        gru_model = Sequential([
            GRU(64, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['GRU'] = gru_model

        # 5. CNN only
        cnn_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',
                   input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['CNN'] = cnn_model

        # 6. Ваша гибридная CNN-LSTM
        hybrid_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',
                   input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            LSTM(128, return_sequences=False),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        hybrid_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['CNN_LSTM_Hybrid'] = hybrid_model

        logger.info(f"Построено {len(self.models)} моделей для сравнения")
        return self.models

    def train_and_evaluate(self, epochs=30, batch_size=32):
        """Обучение и оценка всех моделей"""
        self.results = {}

        for model_name, model in self.models.items():
            logger.info(f"Обучение модели: {model_name}")

            # Замер времени обучения
            start_time = time.time()

            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            training_time = time.time() - start_time

            # Прогнозирование
            y_pred = model.predict(self.X_val, verbose=0)

            # Расчет метрик
            mae = mean_absolute_error(self.y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            r2 = r2_score(self.y_val, y_pred)

            # Сохранение результатов
            self.results[model_name] = {
                'model': model,
                'history': history.history,
                'metrics': {
                    'MAE': float(mae),
                    'RMSE': float(rmse),
                    'R2': float(r2),
                    'training_time': float(training_time)
                },
                'predictions': y_pred.flatten().tolist()
            }

            logger.info(f"Модель {model_name}: MAE={mae:.3f}, RMSE={rmse:.3f}, "
                        f"R²={r2:.3f}, Время={training_time:.1f}с")

        return self.results

    def create_comparison_table(self):
        """Создание сравнительной таблицы"""
        comparison_data = []

        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Архитектура': model_name,
                'MAE': f"{metrics['MAE']:.3f}",
                'RMSE': f"{metrics['RMSE']:.3f}",
                'R²': f"{metrics['R2']:.3f}",
                'Время обучения (с)': f"{metrics['training_time']:.1f}",
                'Параметры': result['model'].count_params()
            })

        df = pd.DataFrame(comparison_data)
        logger.info("Создана сравнительная таблица моделей")
        return df

    def plot_comparison(self, save_path='experiments/results/'):
        """Визуализация результатов сравнения"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Сравнительный анализ архитектур нейронных сетей', fontsize=16)

        # 1. График метрик
        models = list(self.results.keys())
        mae_values = [self.results[m]['metrics']['MAE'] for m in models]
        rmse_values = [self.results[m]['metrics']['RMSE'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width / 2, mae_values, width, label='MAE', color='skyblue')
        bars2 = ax1.bar(x + width / 2, rmse_values, width, label='RMSE', color='lightcoral')
        ax1.set_xlabel('Архитектура')
        ax1.set_ylabel('Ошибка')
        ax1.set_title('Сравнение MAE и RMSE')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()

        # 2. График R²
        r2_values = [self.results[m]['metrics']['R2'] for m in models]

        ax2 = axes[0, 1]
        bars = ax2.bar(models, r2_values, color=['green' if v > 0.9 else 'orange' for v in r2_values])
        ax2.set_xlabel('Архитектура')
        ax2.set_ylabel('R²')
        ax2.set_title('Коэффициент детерминации R²')
        ax2.set_xticklabels(models, rotation=45, ha='right')

        # 3. Время обучения
        time_values = [self.results[m]['metrics']['training_time'] for m in models]

        ax3 = axes[1, 0]
        bars = ax3.bar(models, time_values, color='purple', alpha=0.7)
        ax3.set_xlabel('Архитектура')
        ax3.set_ylabel('Время (секунды)')
        ax3.set_title('Время обучения')
        ax3.set_xticklabels(models, rotation=45, ha='right')

        # 4. Прогнозы лучшей модели
        best_model = min(self.results.items(), key=lambda x: x[1]['metrics']['MAE'])[0]
        predictions = self.results[best_model]['predictions']

        ax4 = axes[1, 1]
        ax4.plot(self.y_val[:50], label='Фактические значения', marker='o', markersize=3)
        ax4.plot(predictions[:50], label='Прогноз', marker='s', markersize=3, alpha=0.7)
        ax4.set_xlabel('Временной шаг')
        ax4.set_ylabel('ОСО (е.Д.)')
        ax4.set_title(f'Прогнозы лучшей модели: {best_model}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        logger.info(f"Графики сохранены в {save_path}")
        plt.show()

        return fig

    def save_results(self, save_path='experiments/results/'):
        """Сохранение всех результатов"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Сохранение таблицы
        df = self.create_comparison_table()
        df.to_csv(os.path.join(save_path, 'comparison_table.csv'), index=False, encoding='utf-8')

        # Сохранение метрик в JSON
        metrics_dict = {}
        for model_name, result in self.results.items():
            metrics_dict[model_name] = result['metrics']

        with open(os.path.join(save_path, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

        # Сохранение графика обучения для каждой модели
        for model_name, result in self.results.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            history = result['history']
            ax1.plot(history['loss'], label='Ошибка обучения')
            ax1.plot(history['val_loss'], label='Ошибка валидации')
            ax1.set_title(f'{model_name} - Функция потерь')
            ax1.set_xlabel('Эпоха')
            ax1.set_ylabel('MSE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(history['mae'], label='MAE обучения')
            ax2.plot(history['val_mae'], label='MAE валидации')
            ax2.set_title(f'{model_name} - Средняя абсолютная ошибка')
            ax2.set_xlabel('Эпоха')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'training_{model_name}.png'), dpi=150)
            plt.close()

        logger.info(f"Все результаты сохранены в {save_path}")


def main():
    """Основная функция для запуска сравнения"""
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК СРАВНИТЕЛЬНОГО АНАЛИЗА АРХИТЕКТУР НЕЙРОСЕТЕЙ")
    logger.info("=" * 60)

    comparator = ModelComparator()

    # Подготовка данных
    comparator.prepare_data()

    # Построение моделей
    comparator.build_models()

    # Обучение и оценка
    results = comparator.train_and_evaluate(epochs=30)

    # Вывод результатов
    df = comparator.create_comparison_table()
    print("\n" + "=" * 80)
    print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА АРХИТЕКТУР НЕЙРОННЫХ СЕТЕЙ")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n" + "=" * 80)

    # Визуализация
    comparator.plot_comparison()

    # Сохранение
    comparator.save_results()

    # Определение лучшей модели
    best_model = min(results.items(), key=lambda x: x[1]['metrics']['MAE'])
    logger.info(f"🏆 Лучшая модель: {best_model[0]} с MAE={best_model[1]['metrics']['MAE']:.3f}")

    return df, results


if __name__ == "__main__":
    df, results = main()