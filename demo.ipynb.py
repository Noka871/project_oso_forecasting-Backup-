# demo.ipynb
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import DataLoader
from src.model import create_model
from src.trainer import train_model, evaluate_model
from src.predictor import make_predictions

# Загрузка данных
print("📊 Загрузка данных...")
loader = DataLoader()
loader.prepare_data()

# Обучение модели
print("🧠 Обучение модели...")
model = create_model()
X_train, y_train = loader.get_train_data()
X_test, y_test = loader.get_test_data()
history = train_model(model, X_train, y_train, X_test, y_test)

# Прогнозирование
print("🔮 Выполнение прогнозов...")
results = make_predictions(model, loader)

# Визуализация
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('History Learning')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results['Actual'], 'o-', label='Actual')
plt.plot(results['Predicted'], 's-', label='Predicted')
plt.title('Predictions vs Actual')
plt.legend()

plt.tight_layout()
plt.show()

print("✅ Demo completed!")