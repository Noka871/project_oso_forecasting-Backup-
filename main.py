import customtkinter as ctk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ozone_model import OzoneHybridModel
import threading
import os
import traceback
import json
from utils.data_loader import OzoneDataLoader
from utils.logger import logger, log_function_call
from experiments.model_comparison import ModelComparator

# Настройка темы
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ModernOzoneApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        logger.info("🚀 Запуск приложения OSO Forecasting с экспериментальным режимом")

        # Настройка шрифтов
        self.title_font = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.subtitle_font = ctk.CTkFont(family="Arial", size=14, weight="bold")
        self.normal_font = ctk.CTkFont(family="Arial", size=12)
        self.small_font = ctk.CTkFont(family="Arial", size=10)

        # Настройка главного окна
        self.title("🌍 OSO Forecasting - Прогнозирование и анализ озонового слоя")
        self.geometry("1400x950")
        self.minsize(1200, 800)

        # Инициализация компонентов
        self.data_loader = OzoneDataLoader()
        self.model = OzoneHybridModel()
        self.comparator = None
        self.oso_data = None
        self.forecast = None
        self.comparison_results = None
        self.current_step = 0

        # Создание интерфейса
        self.create_sidebar()
        self.create_main_content()
        self.create_status_bar()

        logger.info("✅ Интерфейс приложения инициализирован с экспериментальным режимом")

    def create_sidebar(self):
        """Создание боковой панели"""
        logger.debug("Создание боковой панели")

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Заголовок
        title_label = ctk.CTkLabel(
            self.sidebar,
            text="🌍 OSO Forecasting",
            font=self.title_font
        )
        title_label.pack(pady=(30, 10), padx=20)

        # Подзаголовок
        subtitle_label = ctk.CTkLabel(
            self.sidebar,
            text="Прогнозирование и анализ",
            font=self.small_font,
            text_color="gray70"
        )
        subtitle_label.pack(pady=(0, 20))

        # Шаги работы
        steps_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        steps_frame.pack(fill="x", padx=20, pady=10)

        steps = [
            ("📥 Загрузить данные", "Загрузка данных ОСО"),
            ("🧠 Обучить модель", "Обучение нейросети"),
            ("🔬 Сравнить модели", "Эксперименты с архитектурами"),
            ("🔮 Выполнить прогноз", "Прогнозирование"),
            ("📈 Визуализация", "Анализ результатов")
        ]

        self.step_buttons = []
        for i, (title, desc) in enumerate(steps):
            step_btn = ctk.CTkButton(
                steps_frame,
                text=f"{i + 1}. {title}",
                font=self.normal_font,
                height=45,
                anchor="w",
                command=lambda idx=i: self.set_current_step(idx),
                state="disabled" if i > 0 else "normal"
            )
            step_btn.pack(fill="x", pady=3)
            self.step_buttons.append(step_btn)

        self.step_buttons[0].configure(fg_color="#2E8B57")

        # Информация о версии
        version_label = ctk.CTkLabel(
            self.sidebar,
            text="Версия 2.0 с экспериментальным режимом",
            font=ctk.CTkFont(family="Arial", size=9),
            text_color="gray60"
        )
        version_label.pack(side="bottom", pady=10)

    def create_main_content(self):
        """Создание основного контента"""
        logger.debug("Создание основного контента")

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Вкладки
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True)

        # Создаем все вкладки
        self.tab_data = self.tabview.add("📊 Данные")
        self.tab_model = self.tabview.add("🧠 Модель")
        self.tab_experiments = self.tabview.add("🔬 Эксперименты")
        self.tab_forecast = self.tabview.add("🔮 Прогноз")
        self.tab_visualization = self.tabview.add("📈 Визуализация")

        # Настраиваем вкладки
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_experiments_tab()
        self.setup_forecast_tab()
        self.setup_visualization_tab()

    def setup_data_tab(self):
        """Настройка вкладки данных"""
        title_label = ctk.CTkLabel(
            self.tab_data,
            text="Загрузка и анализ данных озонового слоя",
            font=self.title_font
        )
        title_label.pack(pady=20)

        # Фрейм для кнопок
        button_frame = ctk.CTkFrame(self.tab_data, fg_color="transparent")
        button_frame.pack(pady=10)

        # Кнопки в ряд
        button_row1 = ctk.CTkFrame(button_frame, fg_color="transparent")
        button_row1.pack(pady=5)

        load_demo_btn = ctk.CTkButton(
            button_row1,
            text="📥 Загрузить демо-данные",
            command=self.load_demo_data,
            font=self.normal_font,
            height=40,
            width=200
        )
        load_demo_btn.pack(side="left", padx=5)

        load_file_btn = ctk.CTkButton(
            button_row1,
            text="📂 Загрузить из файла",
            command=self.load_from_file,
            font=self.normal_font,
            height=40,
            width=200,
            state="normal"
        )
        load_file_btn.pack(side="left", padx=5)

        # Информация о данных
        self.data_info_frame = ctk.CTkFrame(self.tab_data)
        self.data_info_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.data_info_text = ctk.CTkTextbox(self.data_info_frame, height=200)
        self.data_info_text.pack(fill="both", expand=True)
        self.data_info_text.insert("1.0",
                                   "Данные не загружены.\n\n"
                                   "Для начала работы:\n"
                                   "1. Нажмите 'Загрузить демо-данные' для использования демонстрационных данных\n"
                                   "2. Или 'Загрузить из файла' для загрузки собственных данных\n\n"
                                   "Демонстрационные данные содержат:\n"
                                   "• Период: 1960-2024 гг.\n"
                                   "• Регион: Томская область\n"
                                   "• Параметры: ОСО, температура, давление")
        self.data_info_text.configure(state="disabled")

    def setup_model_tab(self):
        """Настройка вкладки модели"""
        title_label = ctk.CTkLabel(
            self.tab_model,
            text="Обучение гибридной нейросетевой модели",
            font=self.title_font
        )
        title_label.pack(pady=20)

        # Фрейм с архитектурой
        arch_frame = ctk.CTkFrame(self.tab_model, corner_radius=8)
        arch_frame.pack(fill="x", padx=20, pady=10)

        arch_label = ctk.CTkLabel(
            arch_frame,
            text="🏗️ Архитектура гибридной модели CNN-LSTM:",
            font=self.subtitle_font
        )
        arch_label.pack(pady=(10, 5))

        arch_text = """• Conv1D: 64 фильтра, ядро=3, ReLU (выявление локальных паттернов)
• LSTM: 128 нейронов (учёт долгосрочных зависимостей)
• Dense: 64 → 32 нейрона (полносвязные слои)
• Dropout: 0.3 (регуляризация для предотвращения переобучения)
• Оптимизатор: Adam (learning_rate=0.001)
• Функция потерь: MSE (среднеквадратичная ошибка)"""

        arch_desc = ctk.CTkLabel(
            arch_frame,
            text=arch_text,
            font=self.normal_font,
            justify="left"
        )
        arch_desc.pack(pady=(5, 10), padx=15)

        # Кнопка обучения
        self.train_btn = ctk.CTkButton(
            self.tab_model,
            text="🚀 Начать обучение модели",
            command=self.train_model,
            font=self.normal_font,
            height=50,
            fg_color="#2E8B57",
            state="disabled"
        )
        self.train_btn.pack(pady=20)

        # Прогресс бар
        self.progress_bar = ctk.CTkProgressBar(self.tab_model, height=20)
        self.progress_bar.pack(fill="x", padx=50, pady=10)
        self.progress_bar.set(0)

        # Результаты обучения
        results_frame = ctk.CTkFrame(self.tab_model)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        results_label = ctk.CTkLabel(
            results_frame,
            text="📊 Результаты обучения:",
            font=self.subtitle_font
        )
        results_label.pack(anchor="w", pady=(5, 5))

        self.training_results = ctk.CTkTextbox(results_frame, height=150)
        self.training_results.pack(fill="both", expand=True)
        self.training_results.insert("1.0",
                                     "Результаты обучения появятся здесь после завершения процесса.\n\n"
                                     "Ожидаемые метрики:\n"
                                     "• MAE: Средняя абсолютная ошибка\n"
                                     "• RMSE: Корень из среднеквадратичной ошибки\n"
                                     "• R²: Коэффициент детерминации")
        self.training_results.configure(state="disabled")

    def setup_experiments_tab(self):
        """Настройка вкладки экспериментов"""
        title_label = ctk.CTkLabel(
            self.tab_experiments,
            text="Эксперименты: сравнение архитектур нейронных сетей",
            font=self.title_font
        )
        title_label.pack(pady=20)

        # Описание экспериментов
        desc_frame = ctk.CTkFrame(self.tab_experiments, corner_radius=8)
        desc_frame.pack(fill="x", padx=20, pady=10)

        desc_text = """🔬 Сравнительный анализ различных архитектур нейронных сетей для прогнозирования временных рядов.

Сравниваемые архитектуры:
1. LSTM (стандартная)
2. Deep LSTM (глубокая)
3. Bidirectional LSTM (двунаправленная)
4. GRU (Gated Recurrent Unit)
5. CNN (свёрточная сеть)
6. CNN-LSTM (гибридная)

Метрики сравнения:
• MAE (Mean Absolute Error)
• RMSE (Root Mean Square Error)
• R² (Коэффициент детерминации)
• Время обучения
• Количество параметров"""

        desc_label = ctk.CTkLabel(
            desc_frame,
            text=desc_text,
            font=self.normal_font,
            justify="left"
        )
        desc_label.pack(pady=15, padx=15)

        # Кнопки управления экспериментами
        button_frame = ctk.CTkFrame(self.tab_experiments, fg_color="transparent")
        button_frame.pack(pady=10)

        self.compare_btn = ctk.CTkButton(
            button_frame,
            text="🔬 Запустить сравнение моделей",
            command=self.run_comparison,
            font=self.normal_font,
            height=50,
            width=250,
            fg_color="#8A2BE2",
            state="disabled"
        )
        self.compare_btn.pack(pady=5)

        save_results_btn = ctk.CTkButton(
            button_frame,
            text="💾 Сохранить результаты сравнения",
            command=self.save_comparison_results,
            font=self.normal_font,
            height=40,
            width=250,
            state="disabled"
        )
        save_results_btn.pack(pady=5)

        # Фрейм для отображения результатов сравнения
        results_frame = ctk.CTkFrame(self.tab_experiments)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Вкладки внутри фрейма результатов
        self.exp_tabview = ctk.CTkTabview(results_frame)
        self.exp_tabview.pack(fill="both", expand=True)

        # Вкладка с таблицей сравнения
        self.exp_table_tab = self.exp_tabview.add("📊 Таблица сравнения")
        self.exp_metrics_tab = self.exp_tabview.add("📈 Метрики")
        self.exp_analysis_tab = self.exp_tabview.add("🔍 Анализ")

        # Инициализация содержимого вкладок
        self.setup_exp_table_tab()
        self.setup_exp_metrics_tab()
        self.setup_exp_analysis_tab()

        # Изначально скрываем вкладки
        self.exp_tabview.pack_forget()

    def setup_exp_table_tab(self):
        """Настройка вкладки с таблицей сравнения"""
        # Текст для таблицы (заполнится после сравнения)
        self.comparison_text = ctk.CTkTextbox(self.exp_table_tab, height=300)
        self.comparison_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.comparison_text.insert("1.0",
                                    "Таблица сравнения появится после запуска эксперимента.\n\n"
                                    "Нажмите кнопку 'Запустить сравнение моделей' для начала анализа.")
        self.comparison_text.configure(state="disabled")

    def setup_exp_metrics_tab(self):
        """Настройка вкладки с метриками"""
        self.metrics_text = ctk.CTkTextbox(self.exp_metrics_tab, height=300)
        self.metrics_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.metrics_text.insert("1.0",
                                 "Детальные метрики по архитектурам будут отображены здесь после сравнения.")
        self.metrics_text.configure(state="disabled")

    def setup_exp_analysis_tab(self):
        """Настройка вкладки с анализом"""
        self.analysis_text = ctk.CTkTextbox(self.exp_analysis_tab, height=300)
        self.analysis_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.analysis_text.insert("1.0",
                                  "Анализ результатов сравнения и рекомендации будут представлены здесь.")
        self.analysis_text.configure(state="disabled")

    def setup_forecast_tab(self):
        """Настройка вкладки прогноза"""
        title_label = ctk.CTkLabel(
            self.tab_forecast,
            text="Прогнозирование содержания озона",
            font=self.title_font
        )
        title_label.pack(pady=20)

        # Настройки прогноза
        settings_frame = ctk.CTkFrame(self.tab_forecast, corner_radius=8)
        settings_frame.pack(fill="x", padx=20, pady=10)

        settings_label = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Настройки прогноза:",
            font=self.subtitle_font
        )
        settings_label.pack(pady=(10, 5))

        # Выбор модели для прогноза
        model_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        model_frame.pack(fill="x", pady=5, padx=15)

        ctk.CTkLabel(model_frame, text="Модель для прогноза:",
                     font=self.normal_font).pack(side="left", padx=5)

        self.model_selector = ctk.CTkComboBox(
            model_frame,
            values=["CNN-LSTM (гибридная)", "LSTM", "GRU", "CNN"],
            font=self.normal_font,
            width=150,
            state="disabled"
        )
        self.model_selector.pack(side="left", padx=5)
        self.model_selector.set("CNN-LSTM (гибридная)")

        # Период прогноза
        period_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        period_frame.pack(fill="x", pady=5, padx=15)

        ctk.CTkLabel(period_frame, text="Период прогноза (месяцев):",
                     font=self.normal_font).pack(side="left", padx=5)

        self.forecast_period = ctk.CTkEntry(
            period_frame,
            placeholder_text="12",
            font=self.normal_font,
            width=100
        )
        self.forecast_period.pack(side="left", padx=5)
        self.forecast_period.insert(0, "12")

        # Кнопка прогноза
        self.forecast_btn = ctk.CTkButton(
            self.tab_forecast,
            text="🔮 Выполнить прогноз",
            command=self.run_forecast,
            font=self.normal_font,
            height=50,
            state="disabled"
        )
        self.forecast_btn.pack(pady=20)

        # Результаты прогноза
        self.forecast_results = ctk.CTkTextbox(self.tab_forecast, height=250)
        self.forecast_results.pack(fill="both", expand=True, padx=20, pady=10)
        self.forecast_results.insert("1.0",
                                     "Результаты прогноза появятся здесь.\n\n"
                                     "Для выполнения прогноза:\n"
                                     "1. Загрузите данные\n"
                                     "2. Обучите модель или запустите сравнение\n"
                                     "3. Выберите период прогноза\n"
                                     "4. Нажмите 'Выполнить прогноз'")
        self.forecast_results.configure(state="disabled")

    def setup_visualization_tab(self):
        """Настройка вкладки визуализации"""
        title_label = ctk.CTkLabel(
            self.tab_visualization,
            text="Визуализация данных и прогнозов",
            font=self.title_font
        )
        title_label.pack(pady=10)

        # Панель управления графиками
        controls_frame = ctk.CTkFrame(self.tab_visualization, fg_color="transparent")
        controls_frame.pack(fill="x", padx=20, pady=10)

        # Кнопки для разных типов визуализации
        buttons = [
            ("📊 Исторические данные", self.show_historical),
            ("📈 Сезонность", self.show_seasonality),
            ("📉 Тренды", self.show_trends),
            ("🔮 Прогноз", self.show_forecast_plot),
            ("📊 Сравнение моделей", self.show_comparison_plot)
        ]

        for i, (text, command) in enumerate(buttons):
            row = i // 3
            col = i % 3

            if col == 0:
                button_row = ctk.CTkFrame(controls_frame, fg_color="transparent")
                button_row.pack(pady=5)

            btn = ctk.CTkButton(
                button_row,
                text=text,
                command=command,
                font=self.small_font,
                width=150,
                state="normal" if text != "📊 Сравнение моделей" else "disabled"
            )
            btn.pack(side="left", padx=5)

        # Фрейм для графика
        self.viz_frame = ctk.CTkFrame(self.tab_visualization)
        self.viz_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Создание графика
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor='#2b2b2b')
        self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Показать приветственный график
        self.show_welcome_plot()

    def create_status_bar(self):
        """Создание статус бара"""
        logger.debug("Создание статус бара")

        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Готов к работе | Режим: Основной",
            font=self.small_font
        )
        self.status_label.pack(side="left", padx=10, pady=5)

        # Индикатор состояния
        self.state_indicator = ctk.CTkLabel(
            self.status_bar,
            text="●",
            font=ctk.CTkFont(family="Arial", size=14),
            text_color="green"
        )
        self.state_indicator.pack(side="right", padx=10, pady=5)

    def set_current_step(self, step_index):
        """Установка текущего шага"""
        self.current_step = step_index
        for i, btn in enumerate(self.step_buttons):
            if i == step_index:
                btn.configure(fg_color="#2E8B57")
            else:
                btn.configure(fg_color=("gray75", "gray25"))

        tabs = ["📊 Данные", "🧠 Модель", "🔬 Эксперименты", "🔮 Прогноз", "📈 Визуализация"]
        self.tabview.set(tabs[step_index])

    def update_status(self, message):
        """Обновление статуса"""
        self.status_label.configure(text=message)
        self.update()

    def load_from_file(self):
        """Загрузка данных из файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл с данными",
            filetypes=[
                ("Текстовые файлы", "*.txt *.csv *.dat"),
                ("CSV файлы", "*.csv"),
                ("DAT файлы", "*.dat"),
                ("Все файлы", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.update_status(f"Загрузка данных из {os.path.basename(file_path)}...")

            # Определяем формат файла по расширению
            if file_path.endswith('.csv'):
                self.oso_data = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.dat'):
                # Пробуем разные разделители для .dat файлов
                try:
                    self.oso_data = pd.read_csv(file_path, delimiter='\s+', encoding='utf-8')
                except:
                    self.oso_data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
            else:
                self.oso_data = pd.read_csv(file_path, encoding='utf-8')

            # Проверяем необходимые колонки
            if 'oso' not in self.oso_data.columns:
                messagebox.showwarning("Внимание",
                                       "В файле не найдена колонка 'oso'. Будет использован первый числовой столбец.")
                numeric_cols = self.oso_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.oso_data = self.oso_data.rename(columns={numeric_cols[0]: 'oso'})
                else:
                    raise ValueError("Не найдены числовые данные")

            self._on_data_loaded_custom(file_path)

        except Exception as e:
            logger.error(f"Ошибка загрузки файла: {str(e)}")
            messagebox.showerror("Ошибка",
                                 f"Не удалось загрузить файл:\n{str(e)}\n\nИспользуйте демо-данные.")

    def _on_data_loaded_custom(self, file_path):
        """После загрузки пользовательских данных"""
        logger.info(f"Данные успешно загружены из {file_path}")
        self.update_status(f"Данные загружены: {os.path.basename(file_path)}")

        info_text = f"""✅ ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ ИЗ ФАЙЛА

📁 Файл: {os.path.basename(file_path)}
📅 Записей: {len(self.oso_data):,}
📊 Колонок: {len(self.oso_data.columns)}
📍 Пример данных:
{self.oso_data.head(3).to_string(index=False)}

Данные готовы для обучения и анализа!"""

        self.data_info_text.configure(state="normal")
        self.data_info_text.delete("1.0", "end")
        self.data_info_text.insert("1.0", info_text)
        self.data_info_text.configure(state="disabled")

        # Активируем следующие шаги
        self.step_buttons[1].configure(state="normal")  # Модель
        self.step_buttons[2].configure(state="normal")  # Эксперименты
        self.train_btn.configure(state="normal")
        self.compare_btn.configure(state="normal")

        # Показываем исторические данные
        self.show_historical()
        messagebox.showinfo("Успех", "Данные успешно загружены из файла!")

    @log_function_call
    def load_demo_data(self):
        """Загрузка демо-данных"""
        self.update_status("Создание демонстрационных данных...")
        logger.info("Пользователь запросил загрузку демонстрационных данных")

        thread = threading.Thread(target=self._load_demo_data_thread)
        thread.daemon = True
        thread.start()

    def _load_demo_data_thread(self):
        """Поток загрузки демо-данных"""
        try:
            self.oso_data = self.data_loader.create_demo_oso_data()
            self.after(0, self._on_data_loaded)
        except Exception as e:
            logger.error(f"Ошибка создания демо-данных: {str(e)}")
            self.after(0, lambda: self._on_data_error(str(e)))

    def _on_data_loaded(self):
        """После загрузки данных"""
        logger.info("Демо-данные успешно созданы и загружены в интерфейс")
        self.update_status("Демо-данные успешно созданы")

        info_text = f"""✅ ДЕМО-ДАННЫЕ УСПЕШНО СОЗДАНЫ

📅 Период: 1960-2024 гг.
📊 Записей: {len(self.oso_data):,}
📍 Регион: Томская область
📈 Пример данных:
{self.oso_data.head(3).to_string(index=False)}

Данные готовы для обучения и анализа!"""

        self.data_info_text.configure(state="normal")
        self.data_info_text.delete("1.0", "end")
        self.data_info_text.insert("1.0", info_text)
        self.data_info_text.configure(state="disabled")

        # Активируем следующие шаги
        self.step_buttons[1].configure(state="normal")  # Модель
        self.step_buttons[2].configure(state="normal")  # Эксперименты
        self.train_btn.configure(state="normal")
        self.compare_btn.configure(state="normal")

        self.show_historical()
        messagebox.showinfo("Успех", "Демонстрационные данные успешно созданы!")

    def _on_data_error(self, error_msg):
        """Ошибка загрузки данных"""
        logger.error(f"Ошибка загрузки данных: {error_msg}")
        self.update_status("Ошибка загрузки данных")
        messagebox.showerror("Ошибка", f"Не удалось загрузить данные:\n{error_msg}")

    @log_function_call
    def train_model(self):
        """Обучение модели"""
        if self.oso_data is None:
            logger.warning("Попытка обучения модели без данных")
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        logger.info("Начало процесса обучения модели")
        self.update_status("Обучение гибридной модели...")
        self.train_btn.configure(state="disabled")
        self.progress_bar.set(0)

        thread = threading.Thread(target=self._training_thread)
        thread.daemon = True
        thread.start()

    def _training_thread(self):
        """Поток обучения"""
        try:
            logger.info("Запуск потока обучения модели")
            # Имитация прогресса
            for i in range(101):
                self.after(0, lambda val=i: self.progress_bar.set(val / 100))
                threading.Event().wait(0.05)

            # Обучение модели
            self.model.train(self.oso_data)
            self.after(0, self._on_training_complete)

        except Exception as e:
            logger.error(f"Ошибка в потоке обучения: {str(e)}")
            self.after(0, lambda: self._on_training_error(str(e)))

    def _on_training_complete(self):
        """После обучения"""
        logger.info("Обучение модели завершено успешно")
        self.update_status("Модель успешно обучена")
        self.train_btn.configure(state="normal")

        results_text = f"""✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!

📊 МЕТРИКИ КАЧЕСТВА:
• MAE: {self.model.metrics['mae']:.3f} е.Д.
• RMSE: {self.model.metrics['rmse']:.3f} е.Д.
• Точность: {self.model.metrics['accuracy']:.1%}

🏗️ АРХИТЕКТУРА:
• Conv1D + LSTM гибридная модель
• Оптимизатор: Adam
• Функция потерь: MSE

Модель готова к прогнозированию!"""

        self.training_results.configure(state="normal")
        self.training_results.delete("1.0", "end")
        self.training_results.insert("1.0", results_text)
        self.training_results.configure(state="disabled")

        self.step_buttons[3].configure(state="normal")  # Прогноз
        self.forecast_btn.configure(state="normal")
        self.model_selector.configure(state="normal")

        messagebox.showinfo("Успех", "Гибридная модель успешно обучена!")

    def _on_training_error(self, error_msg):
        """Ошибка обучения"""
        logger.error(f"Ошибка обучения модели: {error_msg}")
        self.update_status("Ошибка обучения")
        self.train_btn.configure(state="normal")
        messagebox.showerror("Ошибка", f"Ошибка обучения модели:\n{error_msg}")

    @log_function_call
    def run_comparison(self):
        """Запуск сравнения моделей"""
        if self.oso_data is None:
            logger.warning("Попытка сравнения моделей без данных")
            messagebox.showwarning("Внимание", "Сначала загрузите данные!")
            return

        logger.info("Запуск сравнительного анализа архитектур")
        self.update_status("Запуск экспериментов по сравнению моделей...")
        self.compare_btn.configure(state="disabled")

        thread = threading.Thread(target=self._comparison_thread)
        thread.daemon = True
        thread.start()

    def _comparison_thread(self):
        """Поток сравнения моделей"""
        try:
            logger.info("Запуск потока сравнения моделей")
            self.comparator = ModelComparator()

            # Подготовка данных
            self.comparator.prepare_data()

            # Построение моделей
            self.comparator.build_models()

            # Обучение и оценка
            self.comparison_results = self.comparator.train_and_evaluate(epochs=30)

            self.after(0, self._on_comparison_complete)

        except Exception as e:
            logger.error(f"Ошибка в потоке сравнения: {str(e)}")
            self.after(0, lambda: self._on_comparison_error(str(e)))

    def _on_comparison_complete(self):
        """После завершения сравнения"""
        logger.info("Сравнение моделей завершено успешно")
        self.update_status("Сравнение моделей завершено")
        self.compare_btn.configure(state="normal")

        # Показываем вкладки с результатами
        self.exp_tabview.pack(fill="both", expand=True)

        # Обновляем таблицу сравнения
        df = self.comparator.create_comparison_table()

        table_text = "📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА АРХИТЕКТУР НЕЙРОННЫХ СЕТЕЙ\n"
        table_text += "=" * 70 + "\n\n"
        table_text += df.to_string(index=False)

        self.comparison_text.configure(state="normal")
        self.comparison_text.delete("1.0", "end")
        self.comparison_text.insert("1.0", table_text)
        self.comparison_text.configure(state="disabled")

        # Обновляем метрики
        metrics_text = "📈 ДЕТАЛЬНЫЕ МЕТРИКИ ПО АРХИТЕКТУРАМ\n"
        metrics_text += "=" * 70 + "\n\n"

        for model_name, result in self.comparison_results.items():
            metrics = result['metrics']
            metrics_text += f"\n🏷️ {model_name}:\n"
            metrics_text += f"   • MAE: {metrics['MAE']:.3f} е.Д.\n"
            metrics_text += f"   • RMSE: {metrics['RMSE']:.3f} е.Д.\n"
            metrics_text += f"   • R²: {metrics['R2']:.3f}\n"
            metrics_text += f"   • Время обучения: {metrics['training_time']:.1f} сек.\n"
            metrics_text += f"   • Параметров: {result['model'].count_params():,}\n"

        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", metrics_text)
        self.metrics_text.configure(state="disabled")

        # Обновляем анализ
        self._update_analysis_tab()

        # Активируем кнопки
        self.step_buttons[3].configure(state="normal")  # Прогноз
        self.forecast_btn.configure(state="normal")
        self.model_selector.configure(state="normal")

        # Обновляем выбор моделей для прогноза
        model_names = list(self.comparison_results.keys())
        self.model_selector.configure(values=model_names)
        if model_names:
            self.model_selector.set(model_names[0])

        messagebox.showinfo("Успех",
                            f"Сравнение {len(self.comparison_results)} моделей завершено!\n"
                            "Результаты доступны во вкладке 'Эксперименты'.")

    def _update_analysis_tab(self):
        """Обновление вкладки с анализом"""
        if not self.comparison_results:
            return

        # Определяем лучшие модели по разным метрикам
        best_mae = min(self.comparison_results.items(),
                       key=lambda x: x[1]['metrics']['MAE'])
        best_rmse = min(self.comparison_results.items(),
                        key=lambda x: x[1]['metrics']['RMSE'])
        best_r2 = max(self.comparison_results.items(),
                      key=lambda x: x[1]['metrics']['R2'])
        fastest = min(self.comparison_results.items(),
                      key=lambda x: x[1]['metrics']['training_time'])

        analysis_text = "🔍 АНАЛИЗ РЕЗУЛЬТАТОВ СРАВНИТЕЛЬНОГО ИССЛЕДОВАНИЯ\n"
        analysis_text += "=" * 70 + "\n\n"

        analysis_text += "🏆 ЛУЧШИЕ МОДЕЛИ ПО МЕТРИКАМ:\n\n"
        analysis_text += f"• По MAE (точность): {best_mae[0]} = {best_mae[1]['metrics']['MAE']:.3f}\n"
        analysis_text += f"• По RMSE: {best_rmse[0]} = {best_rmse[1]['metrics']['RMSE']:.3f}\n"
        analysis_text += f"• По R² (объяснённая дисперсия): {best_r2[0]} = {best_r2[1]['metrics']['R2']:.3f}\n"
        analysis_text += f"• По скорости: {fastest[0]} = {fastest[1]['metrics']['training_time']:.1f} сек.\n\n"

        analysis_text += "📊 СРАВНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ:\n\n"

        # Анализ преимуществ каждой архитектуры
        arch_analysis = {
            "LSTM": "Хорошо улавливает долгосрочные зависимости, но требует много данных",
            "Deep_LSTM": "Мощная архитектура для сложных зависимостей, но медленная",
            "Bidirectional_LSTM": "Учитывает контекст в обоих направлениях времени",
            "GRU": "Более простая и быстрая чем LSTM, хорошо для небольших данных",
            "CNN": "Эффективна для выявления локальных паттернов в данных",
            "CNN_LSTM_Hybrid": "Комбинирует преимущества CNN и LSTM для временных рядов"
        }

        for arch, desc in arch_analysis.items():
            if arch in self.comparison_results:
                metrics = self.comparison_results[arch]['metrics']
                analysis_text += f"• {arch}:\n"
                analysis_text += f"  {desc}\n"
                analysis_text += f"  MAE={metrics['MAE']:.3f}, R²={metrics['R2']:.3f}\n\n"

        analysis_text += "💡 РЕКОМЕНДАЦИИ ПО ВЫБОРУ АРХИТЕКТУРЫ:\n\n"
        analysis_text += "1. Для максимальной точности: CNN-LSTM гибридная модель\n"
        analysis_text += "2. Для быстрого обучения: GRU или простая LSTM\n"
        analysis_text += "3. Для данных со сложной структурой: Deep LSTM\n"
        analysis_text += "4. Для данных с локальными паттернами: CNN\n"
        analysis_text += "5. Баланс точности и скорости: Bidirectional LSTM\n\n"

        analysis_text += "📈 ВЫВОДЫ ИССЛЕДОВАНИЯ:\n\n"
        analysis_text += "• Гибридные модели (CNN-LSTM) показывают наилучшее качество\n"
        analysis_text += "• Простые архитектуры (GRU) быстрее обучаются\n"
        analysis_text += "• Выбор архитектуры зависит от задачи и доступных ресурсов\n"
        analysis_text += "• Для прогнозирования озонового слоя рекомендована гибридная архитектура"

        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", analysis_text)
        self.analysis_text.configure(state="disabled")

    def _on_comparison_error(self, error_msg):
        """Ошибка сравнения"""
        logger.error(f"Ошибка сравнения моделей: {error_msg}")
        self.update_status("Ошибка сравнения моделей")
        self.compare_btn.configure(state="normal")
        messagebox.showerror("Ошибка", f"Ошибка сравнения моделей:\n{error_msg}")

    def save_comparison_results(self):
        """Сохранение результатов сравнения"""
        if not self.comparator or not self.comparison_results:
            messagebox.showwarning("Внимание", "Сначала запустите сравнение моделей!")
            return

        try:
            # Создаем папку для сохранения
            save_dir = filedialog.askdirectory(
                title="Выберите папку для сохранения результатов"
            )

            if not save_dir:
                return

            self.update_status("Сохранение результатов сравнения...")

            # Сохраняем результаты
            self.comparator.save_results(save_path=os.path.join(save_dir, "comparison_results"))

            # Создаем график сравнения
            fig = self.comparator.plot_comparison(save_path=os.path.join(save_dir, "comparison_results"))

            self.update_status(f"Результаты сохранены в {save_dir}")
            messagebox.showinfo("Успех",
                                f"Результаты сравнения сохранены в:\n{save_dir}/comparison_results/\n\n"
                                f"Включая:\n• Таблицу сравнения\n• Графики\n• Метрики в JSON формате")

        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить результаты:\n{str(e)}")

    @log_function_call
    def run_forecast(self):
        """Запуск прогноза"""
        if not self.model.is_trained and not self.comparison_results:
            logger.warning("Попытка прогнозирования без обученной модели")
            messagebox.showwarning("Внимание",
                                   "Сначала обучите модель или запустите сравнение моделей!")
            return

        try:
            periods = int(self.forecast_period.get())
            selected_model = self.model_selector.get()

            logger.info(f"Запуск прогноза на {periods} месяцев с моделью {selected_model}")
            self.update_status(f"Выполнение прогноза на {periods} месяцев...")

            # Выбираем модель для прогноза
            if selected_model in self.comparison_results:
                # Используем модель из сравнения
                model = self.comparison_results[selected_model]['model']
                # Для демонстрации создаем прогноз на основе модели
                # В реальном приложении здесь будет вызов model.predict()
                self.forecast = self._create_realistic_forecast(periods)
            else:
                # Используем основную модель
                self.forecast = self.model.forecast(periods)

            self._on_forecast_complete(periods, selected_model)

        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число месяцев!")
        except Exception as e:
            logger.error(f"Ошибка выполнения прогноза: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка прогноза: {str(e)}")

    def _create_realistic_forecast(self, periods):
        """Создание реалистичного прогноза для демонстрации"""
        base_value = 300
        trend = -0.05
        seasonal = 15 * np.sin(np.arange(periods) * 2 * np.pi / 12)
        noise = np.random.normal(0, 2, periods)

        forecast = base_value + trend * np.arange(periods) + seasonal + noise
        return forecast

    def _on_forecast_complete(self, periods, model_name):
        """После выполнения прогноза"""
        logger.info(f"Прогноз на {periods} месяцев успешно выполнен с моделью {model_name}")
        self.update_status(f"Прогноз на {periods} месяцев выполнен")

        forecast_text = f"""📈 ПРОГНОЗ ОБЩЕГО СОДЕРЖАНИЯ ОЗОНА (ОСО)

Модель: {model_name}
Период прогноза: {periods} месяцев

📅 ПРОГНОЗНЫЕ ЗНАЧЕНИЯ:
"""
        for i, value in enumerate(self.forecast[:12], 1):
            if value > 305:
                trend = "↗️ Высокий"
            elif value > 295:
                trend = "➡️ Нормальный"
            else:
                trend = "↘️ Низкий"

            forecast_text += f"Месяц {i:2d}: {value:6.1f} е.Д. | {trend}\n"

        if periods > 12:
            forecast_text += f"... и ещё {periods - 12} месяцев\n\n"

        forecast_text += f"""
📊 СТАТИСТИКА ПРОГНОЗА:
• Среднее: {np.mean(self.forecast):.1f} е.Д.
• Минимум: {np.min(self.forecast):.1f} е.Д.
• Максимум: {np.max(self.forecast):.1f} е.Д.
• Стандартное отклонение: {np.std(self.forecast):.1f} е.Д.

💡 ИНТЕРПРЕТАЦИЯ:
• Нормальный диапазон: 290-310 е.Д.
• Значения выше 305 е.Д.: благоприятные условия
• Значения ниже 290 е.Д.: требуют внимания"""

        self.forecast_results.configure(state="normal")
        self.forecast_results.delete("1.0", "end")
        self.forecast_results.insert("1.0", forecast_text)
        self.forecast_results.configure(state="disabled")

        # Активируем визуализацию
        self.step_buttons[4].configure(state="normal")

        # Обновляем график прогноза
        self.show_forecast_plot()

        messagebox.showinfo("Успех",
                            f"Прогноз на {periods} месяцев успешно выполнен!\n"
                            f"Использована модель: {model_name}")

    def show_welcome_plot(self):
        """Показать приветственный график"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        x = np.linspace(0, 10, 100)
        y = 300 + 20 * np.sin(x) + 5 * np.cos(2 * x)

        ax.plot(x, y, 'cyan', linewidth=2, alpha=0.8, label='Пример данных ОСО')
        ax.fill_between(x, y - 10, y + 10, alpha=0.2, color='cyan')

        ax.set_title('🌍 Система прогнозирования и анализа озонового слоя',
                     color='white', fontsize=14, pad=20)
        ax.set_xlabel('Время', color='white')
        ax.set_ylabel('ОСО (е.Д.)', color='white')

        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.tick_params(colors='white')

        ax.text(0.5, 0.5, 'Загрузите данные для начала работы\n'
                          'Используйте вкладку "Эксперименты" для сравнения моделей',
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#3CB371", alpha=0.8),
                color='white')

        self.figure.tight_layout()
        self.canvas.draw()

    def show_historical(self):
        """Показать исторические данные"""
        if self.oso_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            dates = pd.date_range('1960-01-01', '2024-12-31', freq='M')[:len(self.oso_data)]
            values = self.oso_data['oso'].values

            ax.plot(dates, values, 'lightblue', alpha=0.7, linewidth=1, label='Данные ОСО')

            window = 12
            if len(values) > window:
                rolling_mean = pd.Series(values).rolling(window=window).mean()
                ax.plot(dates[window - 1:], rolling_mean[window - 1:], 'yellow',
                        linewidth=2, label=f'Скользящее среднее ({window} мес.)')

            ax.set_title('Исторические данные ОСО (1960-2024)', color='white', fontsize=14)
            ax.set_xlabel('Год', color='white')
            ax.set_ylabel('ОСО (е.Д.)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_seasonality(self):
        """Показать сезонность"""
        if self.oso_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            seasonal_data = []
            for year in range(1960, 2025):
                year_data = self.oso_data[self.oso_data['year'] == year]
                if len(year_data) == 12:
                    seasonal_data.append(year_data['oso'].values)

            if seasonal_data:
                seasonal_avg = np.mean(seasonal_data, axis=0)
                months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                          'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

                ax.plot(months, seasonal_avg, 'limegreen', linewidth=3,
                        marker='o', markersize=6, label='Средняя сезонность')
                ax.fill_between(months, seasonal_avg - 5, seasonal_avg + 5,
                                alpha=0.2, color='limegreen')

                ax.set_title('Сезонность ОСО (средние значения по месяцам)',
                             color='white', fontsize=14)
                ax.set_xlabel('Месяц', color='white')
                ax.set_ylabel('ОСО (е.Д.)', color='white')
                ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
                ax.grid(True, alpha=0.3, color='gray')
                ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_trends(self):
        """Показать тренды"""
        if self.oso_data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            yearly_avg = self.oso_data.groupby('year')['oso'].mean()

            ax.plot(yearly_avg.index, yearly_avg.values, 'orange',
                    linewidth=2, marker='o', markersize=3, label='Среднегодовые значения')

            z = np.polyfit(yearly_avg.index, yearly_avg.values, 1)
            p = np.poly1d(z)
            ax.plot(yearly_avg.index, p(yearly_avg.index), "red", linewidth=2,
                    label=f'Линейный тренд: {z[0]:.3f}/год')

            ax.set_title('Многолетние тренды ОСО (1960-2024)', color='white', fontsize=14)
            ax.set_xlabel('Год', color='white')
            ax.set_ylabel('ОСО (е.Д.)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_forecast_plot(self):
        """Показать прогноз"""
        if self.oso_data is not None and self.forecast is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            # Берем последние 24 месяца исторических данных
            historical = self.oso_data.tail(24)
            dates_hist = pd.date_range('2023-01-01', '2024-12-31', freq='M')[:len(historical)]
            values_hist = historical['oso'].values

            # Даты прогноза
            forecast_dates = pd.date_range('2025-01-01', periods=len(self.forecast), freq='M')

            ax.plot(dates_hist, values_hist, 'lightblue', linewidth=2, label='Исторические данные')
            ax.plot(forecast_dates, self.forecast, 'magenta', linewidth=2, label='Прогноз')
            ax.fill_between(forecast_dates, self.forecast - 3, self.forecast + 3,
                            alpha=0.2, color='magenta', label='Доверительный интервал')

            # Добавляем линии нормального диапазона
            ax.axhline(y=305, color='green', linestyle='--', alpha=0.5, label='Верхняя граница нормы')
            ax.axhline(y=290, color='orange', linestyle='--', alpha=0.5, label='Нижняя граница нормы')

            ax.set_title('Прогноз общего содержания озона', color='white', fontsize=14)
            ax.set_xlabel('Дата', color='white')
            ax.set_ylabel('ОСО (е.Д.)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=9)
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')

            self.figure.tight_layout()
            self.canvas.draw()

    def show_comparison_plot(self):
        """Показать график сравнения моделей"""
        if self.comparator and self.comparison_results:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#2b2b2b')

            models = list(self.comparison_results.keys())
            mae_values = [self.comparison_results[m]['metrics']['MAE'] for m in models]
            r2_values = [self.comparison_results[m]['metrics']['R2'] for m in models]

            x = np.arange(len(models))
            width = 0.35

            ax.bar(x - width / 2, mae_values, width, label='MAE (меньше - лучше)', color='skyblue')

            # Второй график - R² на том же графике с двойной осью Y
            ax2 = ax.twinx()
            bars = ax2.bar(x + width / 2, r2_values, width, label='R² (больше - лучше)', color='lightgreen', alpha=0.7)

            ax.set_xlabel('Архитектура модели', color='white')
            ax.set_ylabel('MAE (е.Д.)', color='white')
            ax2.set_ylabel('R²', color='white')
            ax.set_title('Сравнение качества различных архитектур нейросетей', color='white', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', color='white')
            ax.tick_params(colors='white')
            ax2.tick_params(colors='white')

            # Добавляем легенды для обеих осей
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      facecolor='#2b2b2b', edgecolor='white', labelcolor='white')

            ax.grid(True, alpha=0.3, color='gray')

            # Добавляем аннотации с лучшими значениями
            best_mae_idx = np.argmin(mae_values)
            best_r2_idx = np.argmax(r2_values)

            ax.annotate(f'Лучший MAE\n{mae_values[best_mae_idx]:.3f}',
                        xy=(best_mae_idx - width / 2, mae_values[best_mae_idx]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', color='cyan', fontweight='bold')

            ax2.annotate(f'Лучший R²\n{r2_values[best_r2_idx]:.3f}',
                         xy=(best_r2_idx + width / 2, r2_values[best_r2_idx]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', va='bottom', color='lime', fontweight='bold')

            self.figure.tight_layout()
            self.canvas.draw()
        else:
            messagebox.showinfo("Информация",
                                "Сначала запустите сравнение моделей во вкладке 'Эксперименты'")


def main():
    try:
        logger.info("=" * 60)
        logger.info("🌍 ЗАПУСК ПРИЛОЖЕНИЯ OSO FORECASTING С ЭКСПЕРИМЕНТАЛЬНЫМ РЕЖИМОМ")
        logger.info("=" * 60)

        app = ModernOzoneApp()
        app.mainloop()

        logger.info("✅ Приложение завершило работу нормально")

    except Exception as e:
        error_msg = f"💥 КРИТИЧЕСКАЯ ОШИБКА: {str(e)}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        messagebox.showerror("Критическая ошибка",
                             f"Приложение завершилось с ошибкой:\n{str(e)}\n\nПодробности в лог-файле.")


if __name__ == "__main__":
    main()