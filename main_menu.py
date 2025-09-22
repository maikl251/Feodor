import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, filedialog, simpledialog
import os
import json
import threading
import time
import datetime
import logging  # Добавлено для унифицированного логирования
import glob
import tensorflow as tf
from lem1 import LemmatizerProcessor
from vae import TextVAE
from q import QTextNetwork

try:
    from pymorphy2 import MorphAnalyzer
    from morph_singleton import get_morph
except ImportError:
    print("pymorphy2 не установлен. Используется упрощённая лемматизация.")
    MorphAnalyzer = None
    def get_morph():
        return None

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Создание директории logs перед настройкой logging
os.makedirs("logs", exist_ok=True)

# Настройка единого лога
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ConfidenceDialog(tk.Toplevel):
    def __init__(self, parent, initial_value):
        super().__init__(parent)
        self.title("Порог уверенности")
        self.geometry("300x150")
        self.resizable(False, False)
        
        ttk.Label(self, text="Введите порог уверенности (0.00 - 1.00):").pack(pady=10, padx=10)
        
        self.entry_var = tk.StringVar(value=f"{initial_value:.2f}")
        entry = ttk.Entry(self, textvariable=self.entry_var, width=10)
        entry.pack(pady=5)
        entry.bind("<Return>", lambda e: self.validate_and_close())
        
        ttk.Button(self, text="OK", command=self.validate_and_close).pack(pady=10)
        
        self.value = None
        self.grab_set()
        self.focus_set()
        self.wait_window()
    
    def validate_and_close(self):
        value_str = self.entry_var.get().replace(',', '.')
        try:
            value = float(value_str)
            if 0.0 <= value <= 1.0:
                self.value = round(value, 2)  # Округляем до двух знаков для точности 0.01
                self.destroy()
            else:
                messagebox.showerror("Ошибка", "Значение должно быть между 0.00 и 1.00", parent=self)
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение. Введите число (используйте точку или запятую как разделитель).", parent=self)

class SimilarityDialog(tk.Toplevel):
    def __init__(self, parent, initial_value):
        super().__init__(parent)
        self.title("Порог сходства")
        self.geometry("300x150")
        self.resizable(False, False)
        
        ttk.Label(self, text="Введите порог сходства для поиска пары (0.00 - 1.00):").pack(pady=10, padx=10)
        
        self.entry_var = tk.StringVar(value=f"{initial_value:.2f}")
        entry = ttk.Entry(self, textvariable=self.entry_var, width=10)
        entry.pack(pady=5)
        entry.bind("<Return>", lambda e: self.validate_and_close())
        
        ttk.Button(self, text="OK", command=self.validate_and_close).pack(pady=10)
        
        self.value = None
        self.grab_set()
        self.focus_set()
        self.wait_window()
    
    def validate_and_close(self):
        value_str = self.entry_var.get().replace(',', '.')
        try:
            value = float(value_str)
            if 0.0 <= value <= 1.0:
                self.value = round(value, 2)  # Округляем до двух знаков для точности 0.01
                self.destroy()
            else:
                messagebox.showerror("Ошибка", "Значение должно быть между 0.00 и 1.00", parent=self)
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение. Введите число (используйте точку или запятую как разделитель).", parent=self)

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Analysis and Generation System")
        self.root.geometry("1000x800")
        
        self.vae_epochs = 10
        self.vae_batch_size = 32
        self.q_epochs = 5
        self.q_batch_size = 32
        self.temperature = 0.8
        self.queries_per_topic = 5
        self.confidence_threshold = 0.0
        self.similarity_threshold = 0.6  # Новый параметр для порога сходства
        self.loss_threshold = 0.01
        
        self.processing_active = False
        self.processed_files = set()
        
        self.lemmatizer = LemmatizerProcessor()
        self.vae = TextVAE()
        self.q_network = QTextNetwork()
        self.morph = get_morph()
        
        self.create_widgets()
        self.create_initial_files()
        self.load_processed_files()
    
    def create_initial_files(self):
        folders = ["тексты", "vae_model", "q_learning_model"]  # logs уже создана
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        initial_files = {
            "base_corpus.json": [],
            "qa_pairs.json": [],
            "processed_files.json": [],
            "themes_data.json": {}
        }
        for filename, default_data in initial_files.items():
            if not os.path.exists(filename):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, ensure_ascii=False, indent=2)
    
    def safe_execute(self, operation, fallback_value=None, error_message="Ошибка"):
        try:
            return operation()
        except Exception as e:
            self.log_message(f"{error_message}: {str(e)}")
            messagebox.showerror("Ошибка", f"{error_message}: {str(e)}")
            return fallback_value
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Управление")
        
        generation_frame = ttk.Frame(self.notebook)
        self.notebook.add(generation_frame, text="Генерация")
        
        self.setup_control_tab(control_frame)
        self.setup_generation_tab(generation_frame)
        
        self.status_var = tk.StringVar(value="Система готова к работе. Добавьте тексты в папку 'тексты'")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, 
                            relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_control_tab(self, parent):
        params_frame = ttk.LabelFrame(parent, text="Параметры обучения")
        params_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(params_frame, text="VAE Эпохи:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.vae_epochs_var = tk.StringVar(value=str(self.vae_epochs))
        ttk.Entry(params_frame, textvariable=self.vae_epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="VAE Batch Size:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.vae_batch_var = tk.StringVar(value=str(self.vae_batch_size))
        ttk.Entry(params_frame, textvariable=self.vae_batch_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Q Эпохи:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.q_epochs_var = tk.StringVar(value=str(self.q_epochs))
        ttk.Entry(params_frame, textvariable=self.q_epochs_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Q Batch Size:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.q_batch_var = tk.StringVar(value=str(self.q_batch_size))
        ttk.Entry(params_frame, textvariable=self.q_batch_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Температура:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.temp_var = tk.StringVar(value=str(self.temperature))
        ttk.Entry(params_frame, textvariable=self.temp_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Запросов на тему:").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self.queries_per_topic_var = tk.StringVar(value=str(self.queries_per_topic))
        ttk.Entry(params_frame, textvariable=self.queries_per_topic_var, width=10).grid(row=2, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Порог потерь:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.loss_threshold_var = tk.StringVar(value=str(self.loss_threshold))
        ttk.Entry(params_frame, textvariable=self.loss_threshold_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        self.use_early_stopping = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Включить early stopping", variable=self.use_early_stopping).grid(row=3, column=2, columnspan=2, pady=5)
        
        ttk.Button(params_frame, text="Установить порог уверенности", command=self.set_confidence_threshold).grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(params_frame, text="Установить порог сходства", command=self.set_similarity_threshold).grid(row=4, column=2, columnspan=2, pady=10)
        
        control_buttons_frame = ttk.Frame(parent)
        control_buttons_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Button(control_buttons_frame, text="Сканировать корпус", command=self.scan_corpus).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_buttons_frame, text="Обучить VAE", command=self.train_vae).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_buttons_frame, text="Генерировать запросы", command=self.generate_queries).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_buttons_frame, text="Обучить Q-сеть", command=self.train_q).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_buttons_frame, text="Запустить автообработку", command=self.start_auto_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_buttons_frame, text="Остановить автообработку", command=self.stop_auto_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_buttons_frame, text="Выход", command=self.exit_program).pack(side=tk.LEFT, padx=5)  # Добавлена кнопка выхода
        
        self.log_text = scrolledtext.ScrolledText(parent, height=15, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def setup_generation_tab(self, parent):
        ttk.Label(parent, text="Введите запрос:").pack(pady=5)
        self.query_input = scrolledtext.ScrolledText(parent, height=5, width=100)
        self.query_input.pack(pady=5)
        
        ttk.Button(parent, text="Сгенерировать ответ", command=self.generate_response).pack(pady=5)
        
        ttk.Label(parent, text="Результат:").pack(pady=5)
        self.result_text = scrolledtext.ScrolledText(parent, height=15, width=100)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def update_parameters(self):
        try:
            self.vae_epochs = int(self.vae_epochs_var.get())
            self.vae_batch_size = int(self.vae_batch_var.get())
            self.q_epochs = int(self.q_epochs_var.get())
            self.q_batch_size = int(self.q_batch_var.get())
            self.temperature = float(self.temp_var.get())
            self.queries_per_topic = int(self.queries_per_topic_var.get())
            self.loss_threshold = float(self.loss_threshold_var.get())
            return True
        except ValueError:
            messagebox.showerror("Ошибка", "Неверные значения параметров")
            return False
    
    def set_confidence_threshold(self):
        dialog = ConfidenceDialog(self.root, self.confidence_threshold)
        if dialog.value is not None:
            self.confidence_threshold = dialog.value
            self.log_message(f"Порог уверенности установлен на {self.confidence_threshold:.2f}")
    
    def set_similarity_threshold(self):
        dialog = SimilarityDialog(self.root, self.similarity_threshold)
        if dialog.value is not None:
            self.similarity_threshold = dialog.value
            self.log_message(f"Порог сходства установлен на {self.similarity_threshold:.2f}")
    
    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry + "\n"))
        self.root.after(0, lambda: self.log_text.see(tk.END))
        logging.info(log_entry)  # Запись в app.log через logging
        print(log_entry)
    
    def load_processed_files(self):
        if os.path.exists("processed_files.json"):
            try:
                with open("processed_files.json", 'r', encoding='utf-8') as f:
                    self.processed_files = set(json.load(f))
            except:
                self.processed_files = set()
        else:
            self.processed_files = set()
    
    def save_processed_files(self):
        with open("processed_files.json", 'w', encoding='utf-8') as f:
            json.dump(list(self.processed_files), f, ensure_ascii=False, indent=2)
    
    def scan_corpus(self):
        if not self.update_parameters():
            return
        self.log_message("Сканирование корпуса...")
        new_texts = self.vae.scan_for_new_corpora()
        self.processed_files.update(self.vae.processed_files)
        self.save_processed_files()
        self.log_message(f"Найдено новых текстов: {len(new_texts)}, корпус: {len(self.vae.full_corpus)}")
    
    def train_vae(self):
        if not self.update_parameters():
            return
        self.log_message("Запуск обучения VAE...")
        if not self.vae.full_corpus:
            self.log_message("Корпус пуст. Сначала сканируйте тексты.")
            return
        history = self.vae.train_model(
            epochs=self.vae_epochs,
            batch_size=self.vae_batch_size,
            use_early_stopping=self.use_early_stopping.get(),
            loss_threshold=self.loss_threshold
        )
        if history:
            self.log_message("VAE успешно обучен/дообучен")
            if hasattr(history, 'history') and 'loss' in history.history:
                self.log_message(f"Количество эпох: {len(history.history['loss'])}")
                self.log_message(f"Финальные потери: {history.history['loss'][-1]:.4f}")
        else:
            self.log_message("Ошибка обучения VAE")
    
    def generate_queries(self):
        if not self.update_parameters():
            return
        self.log_message("Генерация запросов...")
        if not self.vae.is_trained:
            self.log_message("VAE не обучен. Сначала обучите модель.")
            return
        generated_count = self.vae.generate_queries(self.queries_per_topic, self.temperature)
        self.log_message(f"Сгенерировано пар: {generated_count}")
    
    def train_q(self):
        if not self.update_parameters():
            return
        self.log_message("Обучение Q-сети...")
        history = self.q_network.train(
            epochs=self.q_epochs,
            batch_size=self.q_batch_size,
            use_early_stopping=self.use_early_stopping.get(),
            loss_threshold=self.loss_threshold
        )
        if history:
            self.log_message("Q-сеть успешно обучена")
            if hasattr(history, 'history') and 'loss' in history.history:
                self.log_message(f"Количество эпох: {len(history.history['loss'])}")
                self.log_message(f"Финальные потери: {history.history['loss'][-1]:.4f}")
        else:
            self.log_message("Ошибка обучения Q-сети")
    
    def generate_response(self):
        query = self.query_input.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Предупреждение", "Введите запрос для генерации")
            return
        
        if self.morph:
            lemmatized_query = ' '.join(self.morph.parse(word)[0].normal_form for word in query.lower().split())
        else:
            lemmatized_query = ' '.join(query.lower().split())
        
        self.log_message(f"Генерация ответа на запрос: {lemmatized_query}")
        
        best_pair, similarity_score = self.q_network.find_similar_question(lemmatized_query)
        if best_pair and similarity_score > self.similarity_threshold:  # Используем динамический порог
            self.log_message(f"Найдена близкая пара: {best_pair['question']} -> {best_pair['answer']}")
            seed_sequence = best_pair['answer'].split()
            completed_answer, confidence = self.q_network.generate_text_from_seed(seed_sequence)
            response = completed_answer
        else:
            self.log_message("Близкая пара не найдена, генерация с нуля.")
            response, confidence = self.q_network.generate_text(lemmatized_query)

        self.result_text.delete("1.0", tk.END)
        if confidence < self.confidence_threshold:
            messagebox.showwarning("Предупреждение", "Нет полной уверенности в ответе. Загрузите дополнительные тексты по этой теме.")
            self.result_text.insert(tk.END, f"{response} (Уверенность: {confidence:.2f})")
        else:
            self.result_text.insert(tk.END, response)
        self.log_message(f"Сгенерирован ответ: {response} с уверенностью {confidence:.2f}")
    
    def start_auto_processing(self):
        if self.processing_active:
            return
        
        self.processing_active = True
        self.status_var.set("Автоматическая обработка запущена...")
        threading.Thread(target=self.auto_processing_loop, daemon=True).start()
    
    def stop_auto_processing(self):
        self.processing_active = False
        self.status_var.set("Автоматическая обработка остановлена")
    
    def auto_processing_loop(self):
        consecutive_no_new = 0
        max_consecutive_no_new = 3
        while self.processing_active:
            try:
                if not self.update_parameters():
                    time.sleep(1)
                    continue
                
                self.log_message("Авто: Сканирование и обработка новых файлов...")
                new_texts = self.vae.scan_for_new_corpora()
                self.processed_files.update(self.vae.processed_files)
                self.save_processed_files()
                
                self.log_message(f"Авто: Найдено новых текстов: {len(new_texts)}, корпус: {len(self.vae.full_corpus)}")
                if new_texts and self.vae.full_corpus:
                    consecutive_no_new = 0
                    self.log_message(f"Авто: Обнаружено новых текстов: {len(new_texts)}. Дообучение VAE.")
                    history = self.vae.train_model(
                        epochs=self.vae_epochs,
                        batch_size=self.vae_batch_size,
                        new_texts=new_texts,
                        use_early_stopping=self.use_early_stopping.get(),
                        loss_threshold=self.loss_threshold
                    )
                    
                    self.log_message("Авто: Новые данные добавлены в корпус")
                    with open("themes_data.json", 'r', encoding='utf-8') as f:
                        themes_data = json.load(f)
                    with open("base_corpus.json", 'r', encoding='utf-8') as f:
                        corpus_data = json.load(f)
                    self.log_message(f"Авто: Текстов в базе: {len(corpus_data)}")
                    self.log_message(f"Авто: Тем в базе: {len(themes_data)}")
                    
                    if history and hasattr(history, 'history') and 'loss' in history.history:
                        self.log_message(f"Авто: Количество эпох VAE: {len(history.history['loss'])}")
                        self.log_message(f"Авто: Финальные потери VAE: {history.history['loss'][-1]:.4f}")
                    
                    self.log_message("Авто: Генерация запросов...")
                    generated_count = self.vae.generate_queries(self.queries_per_topic, self.temperature)
                    self.log_message(f"Авто: Сгенерировано пар: {generated_count}")
                    
                    self.log_message("Авто: Дообучение Q-сети...")
                    q_history = self.q_network.train(
                        epochs=self.q_epochs,
                        batch_size=self.q_batch_size,
                        use_early_stopping=self.use_early_stopping.get(),
                        loss_threshold=self.loss_threshold
                    )
                    if q_history and hasattr(q_history, 'history') and 'loss' in q_history.history:
                        self.log_message(f"Авто: Количество эпох Q: {len(q_history.history['loss'])}")
                        self.log_message(f"Авто: Финальные потери Q: {q_history.history['loss'][-1]:.4f}")
                    
                    # Изменено: Завершение автообработки без выхода
                    self.log_message("Авто: Обучение завершено. Теперь можно генерировать тексты во вкладке 'Генерация'.")
                    self.processing_active = False
                    self.status_var.set("Обучение завершено. Перейдите к вкладке 'Генерация'.")
                else:
                    consecutive_no_new += 1
                    self.log_message(f"Авто: Нет новых текстов для обучения (проверьте папку 'тексты'). Цикл #{consecutive_no_new}")
                    if consecutive_no_new >= max_consecutive_no_new:
                        self.log_message("Авто: Достигнут лимит циклов без изменений. Увеличиваем паузу.")
                        time.sleep(30)
                        consecutive_no_new = 0
                    else:
                        time.sleep(5)
                    continue
                
                time.sleep(5)
            except Exception as e:
                self.log_message(f"Авто: Ошибка в цикле: {e}")
                time.sleep(1)
    
    def exit_program(self):
        self.log_message("Выход из программы.")
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()

if __name__ == "__main__":
    main()
