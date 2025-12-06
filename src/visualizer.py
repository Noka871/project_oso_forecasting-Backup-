# src/visualizer.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(results_df, save_path="results/predictions_plot.png"):
    """Визуализация прогнозов"""
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Actual'], label='Реальные значения', marker='o')
    plt.plot(results_df['Predicted'], label='Предсказания', marker='x')
    plt.xlabel('Временные точки')
    plt.ylabel('Значение ОСО')
    plt.title('Сравнение реальных и предсказанных значений')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()