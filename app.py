# app.py
from flask import Flask, render_template, request, jsonify
import base64
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Глобальные переменные для состояния
data_loaded = False
model_trained = False
current_df = None
current_model = None
current_scaler = StandardScaler()
training_history = None


class OzoneForecaster:
    def __init__(self, n_steps=60, n_features=12):
        self.n_steps = n_steps
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def create_synthetic_data(self):
        """Создание реалистичных демонстрационных данных"""
        print("🔧 Создание демонстрационных данных...")
        dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
        n = len(dates)

        # Базовый тренд
        base_trend = np.linspace(320, 310, n)

        # Сезонная компонента
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)

        # Аномалии с 2020 года
        anomaly = np.zeros(n)
        anomaly_mask = (dates >= '2020-01-01')
        anomaly[anomaly_mask] = -8 * np.sin(2 * np.pi * (np.arange(n)[anomaly_mask] % 365.25) / 365.25)

        noise = np.random.normal(0, 5, n)
        ozone_data = base_trend + seasonal + anomaly + noise

        # Создаем DataFrame с основными признаками
        data_dict = {'date': dates, 'ozone': np.clip(ozone_data, 280, 360)}

        # Добавляем дополнительные признаки (температура, давление, влажность)
        for i, feature_name in enumerate(['temperature', 'pressure', 'humidity', 'wind_speed', 'solar_rad']):
            base_value = [15, 1013, 65, 3, 150][i]
            seasonal_var = [10, 20, 15, 5, 50][i]
            feature_data = base_value + seasonal_var * np.sin(2 * np.pi * (np.arange(n) / 365.25 + i / 6))
            data_dict[feature_name] = feature_data + np.random.normal(0, 2, n)

        df = pd.DataFrame(data_dict)
        df.set_index('date', inplace=True)
        return df

    def create_features(self, df, target_column='ozone'):
        """Создание признаков для временного ряда"""
        df_features = df.copy()

        # Временные признаки
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['day_of_year'] = df_features.index.dayofyear
        df_features['week'] = df_features.index.isocalendar().week

        # Тригонометрические признаки для сезонности
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

        # Лаговые признаки
        for lag in [1, 2, 7, 30]:
            df_features[f'{target_column}_lag_{lag}'] = df_features[target_column].shift(lag)

        # Скользящие статистики
        for window in [7, 30]:
            df_features[f'{target_column}_rolling_mean_{window}'] = (
                df_features[target_column].rolling(window=window).mean()
            )
            df_features[f'{target_column}_rolling_std_{window}'] = (
                df_features[target_column].rolling(window=window).std()
            )

        # Удаление строк с NaN
        df_features = df_features.dropna()
        return df_features

    def prepare_sequences(self, X, y, time_steps=1):
        """Подготовка последовательностей для LSTM"""
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    def build_model(self, input_shape):
        """Построение гибридной модели Conv1D + LSTM"""
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model


def plot_to_base64():
    """Конвертирует график в base64 строку"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return image_base64


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/load_data', methods=['POST'])
def load_data():
    """API для загрузки данных"""
    global data_loaded, current_df

    try:
        forecaster = OzoneForecaster()
        current_df = forecaster.create_synthetic_data()
        data_loaded = True

        # Создаем график данных
        plt.figure(figsize=(12, 6))
        plt.plot(current_df.index, current_df['ozone'], 'b-', alpha=0.7, linewidth=1)
        plt.title('📊 Исторические данные содержания озона (2000-2024)')
        plt.xlabel('Год')
        plt.ylabel('ОСО (ед. Добсона)')
        plt.grid(True, alpha=0.3)
        data_plot = plot_to_base64()

        return jsonify({
            'success': True,
            'message': f'✅ Данные загружены! Записей: {len(current_df):,}',
            'plot': data_plot,
            'stats': {
                'period': f"{current_df.index.min().strftime('%Y-%m-%d')} - {current_df.index.max().strftime('%Y-%m-%d')}",
                'mean_ozone': f"{current_df['ozone'].mean():.1f}",
                'min_ozone': f"{current_df['ozone'].min():.1f}",
                'max_ozone': f"{current_df['ozone'].max():.1f}"
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'❌ Ошибка загрузки: {str(e)}'
        })


@app.route('/api/train_model', methods=['POST'])
def train_model():
    """API для обучения модели"""
    global model_trained, current_model, current_scaler, training_history

    if not data_loaded:
        return jsonify({
            'success': False,
            'message': '❌ Сначала загрузите данные!'
        })

    try:
        forecaster = OzoneForecaster()

        # Создание признаков
        df_features = forecaster.create_features(current_df)
        X = df_features.drop(columns=['ozone'])
        y = df_features['ozone']

        # Корректировка размерности
        if X.shape[1] < forecaster.n_features:
            for i in range(X.shape[1], forecaster.n_features):
                X[f'synthetic_feature_{i}'] = np.random.normal(0, 1, len(X))
        elif X.shape[1] > forecaster.n_features:
            X = X.iloc[:, :forecaster.n_features]

        # Разделение данных
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Масштабирование
        X_train_scaled = current_scaler.fit_transform(X_train)

        # Подготовка последовательностей
        X_train_seq, y_train_seq = forecaster.prepare_sequences(
            X_train_scaled, y_train.values, forecaster.n_steps
        )

        # Построение и обучение модели
        input_shape = (forecaster.n_steps, X_train_seq.shape[2])
        current_model = forecaster.build_model(input_shape)

        print("🧠 Начало обучения модели...")
        forecaster.history = current_model.fit(
            X_train_seq, y_train_seq,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        training_history = forecaster.history
        model_trained = True

        # Создаем график обучения
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(forecaster.history.history['loss'], label='Training Loss')
        plt.plot(forecaster.history.history['val_loss'], label='Validation Loss')
        plt.title('История обучения - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(forecaster.history.history['mae'], label='Training MAE')
        plt.plot(forecaster.history.history['val_mae'], label='Validation MAE')
        plt.title('История обучения - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        training_plot = plot_to_base64()

        return jsonify({
            'success': True,
            'message': '✅ Модель успешно обучена! (30 эпох)',
            'plot': training_plot
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'❌ Ошибка обучения: {str(e)}'
        })


@app.route('/api/predict', methods=['POST'])
def predict():
    """API для прогнозирования"""
    global current_df, current_model, current_scaler

    if not data_loaded or not model_trained:
        return jsonify({
            'success': False,
            'message': '❌ Сначала загрузите данные и обучите модель!'
        })

    try:
        forecaster = OzoneForecaster()

        # Создание признаков
        df_features = forecaster.create_features(current_df)
        X = df_features.drop(columns=['ozone'])
        y = df_features['ozone']

        # Корректировка размерности
        if X.shape[1] < forecaster.n_features:
            for i in range(X.shape[1], forecaster.n_features):
                X[f'synthetic_feature_{i}'] = np.random.normal(0, 1, len(X))
        elif X.shape[1] > forecaster.n_features:
            X = X.iloc[:, :forecaster.n_features]

        # Разделение данных
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        # Масштабирование и подготовка последовательностей
        X_test_scaled = current_scaler.transform(X_test)
        X_test_seq, y_test_seq = forecaster.prepare_sequences(
            X_test_scaled, y_test.values, forecaster.n_steps
        )

        # Прогнозирование
        y_pred = current_model.predict(X_test_seq, verbose=0).flatten()

        # Расчет метрик
        mse = mean_squared_error(y_test_seq[:len(y_pred)], y_pred)
        mae = mean_absolute_error(y_test_seq[:len(y_pred)], y_pred)
        rmse = np.sqrt(mse)

        # Создаем график прогнозов
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        test_dates = current_df.index[split_idx + forecaster.n_steps:split_idx + forecaster.n_steps + len(y_pred)]

        plt.plot(test_dates, y_test_seq[:len(y_pred)], 'b-', label='Фактические значения', linewidth=2)
        plt.plot(test_dates, y_pred, 'r--', label='Прогноз модели', linewidth=2)
        plt.title('🎯 Сравнение прогноза с фактическими значениями')
        plt.xlabel('Дата')
        plt.ylabel('ОСО (ед. Добсона)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        errors = y_test_seq[:len(y_pred)] - y_pred
        plt.plot(test_dates, errors, 'g-', alpha=0.7, label='Ошибка прогноза')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.fill_between(test_dates, errors, 0, alpha=0.3, color='green')
        plt.title('📉 Ошибки прогнозирования')
        plt.xlabel('Дата')
        plt.ylabel('Ошибка (ед. Добсона)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        predictions_plot = plot_to_base64()

        return jsonify({
            'success': True,
            'message': f'✅ Прогнозы выполнены! MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}',
            'plot': predictions_plot,
            'metrics': {
                'mse': f"{mse:.2f}",
                'mae': f"{mae:.2f}",
                'rmse': f"{rmse:.2f}"
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'❌ Ошибка прогнозирования: {str(e)}'
        })


@app.route('/api/status', methods=['GET'])
def get_status():
    """API для получения статуса"""
    return jsonify({
        'data_loaded': data_loaded,
        'model_trained': model_trained,
        'data_count': len(current_df) if current_df is not None else 0
    })


if __name__ == '__main__':
    print("🚀 Запуск улучшенного веб-приложения...")
    print("📊 Доступно по адресу: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)