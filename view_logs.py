import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from datetime import datetime
import glob


class LogViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("📋 Просмотр логов OSO Forecasting")
        self.root.geometry("1000x700")

        self.create_widgets()
        self.load_latest_log()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Панель управления
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Обновить", command=self.load_latest_log).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Выбрать файл", command=self.select_log_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Очистить логи", command=self.clear_logs).pack(side=tk.LEFT)

        # Информация о файле
        self.file_info = ttk.Label(control_frame, text="")
        self.file_info.pack(side=tk.RIGHT)

        # Текстовое поле для логов
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=120, height=40)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Статус бар
        self.status_bar = ttk.Label(self.root, text="Готов", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)

    def get_log_files(self):
        """Получение списка лог-файлов"""
        if not os.path.exists('logs'):
            return []
        return sorted(glob.glob('logs/*.log'), reverse=True)

    def load_latest_log(self):
        """Загрузка последнего лог-файла"""
        log_files = self.get_log_files()
        if log_files:
            self.load_log_file(log_files[0])
        else:
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "Лог-файлы не найдены")
            self.file_info.config(text="Файлы не найдены")

    def select_log_file(self):
        """Выбор лог-файла"""
        log_files = self.get_log_files()
        if not log_files:
            tk.messagebox.showwarning("Внимание", "Лог-файлы не найдены")
            return

        # Создаем окно выбора файла
        select_window = tk.Toplevel(self.root)
        select_window.title("Выбор лог-файла")
        select_window.geometry("400x300")

        listbox = tk.Listbox(select_window)
        for file in log_files:
            listbox.insert(tk.END, os.path.basename(file))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def on_select():
            selection = listbox.curselection()
            if selection:
                filename = log_files[selection[0]]
                self.load_log_file(filename)
                select_window.destroy()

        ttk.Button(select_window, text="Выбрать", command=on_select).pack(pady=5)

    def load_log_file(self, filename):
        """Загрузка указанного лог-файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, content)

            # Показываем информацию о файле
            file_size = os.path.getsize(filename)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filename))
            self.file_info.config(text=f"{os.path.basename(filename)} | {file_size / 1024:.1f} KB | {mod_time}")

            self.status_bar.config(text=f"Загружен: {os.path.basename(filename)}")

            # Прокручиваем вниз
            self.log_text.see(tk.END)

        except Exception as e:
            tk.messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def clear_logs(self):
        """Очистка всех логов"""
        if tk.messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить все логи?"):
            log_files = self.get_log_files()
            for file in log_files:
                try:
                    os.remove(file)
                except:
                    pass
            self.load_latest_log()
            self.status_bar.config(text="Логи очищены")


def main():
    root = tk.Tk()
    app = LogViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()