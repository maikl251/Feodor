Автоматизированная система обработки и генерации текстов на основе VAE и Q-сетей
Переключение между CNN и LSTM VAE
По умолчанию используется модель **CNN-VAE**, реализованная в файле `vae.py`.  
Для использования альтернативной модели **LSTM-VAE** (`vae1.py`) выполните следующие действия:
➤ Активация LSTM-VAE:
1. Переименуйте файл `vae.py` → `vae_backup.py`
2. Переименуйте файл `vae1.py` → `vae.py`
 ➤ Возврат к CNN-VAE:
1. Переименуйте файл `vae_backup.py` → `vae.py`
2. Переименуйте текущий `vae.py` (LSTM-версия) → `vae1.py`
**Важно**: Следите за целостностью файлов. Рекомендуется использовать систему контроля версий (например, Git) перед внесением изменений.

Структура проекта

- `main_menu.py` — графический интерфейс (GUI)
- `vae.py` — модель вариационного автоэнкодера на основе CNN (по умолчанию)
- `vae1.py` — модель вариационного автоэнкодера на основе LSTM (альтернатива)
- `q.py` — реализация Q-сети
- `lem1.py` — модуль лемматизации текста
- `morph_singleton.py` — обёртка для `pymorphy2` (морфологический анализатор для русского языка)

Использование

1. Поместите текстовые файлы в папку:  
   `папка - текст/название_темы*/`
2. Запустите приложение командой:  
   ```bash
   python main_menu.py
   ```
3. Выберите режим работы:
   - **Автообучение** — автоматическое обучение без участия пользователя
   - **Обучение по шагам** — поэтапное управление процессом обучения
4. После завершения обучения перейдите на вкладку **"Генерация"**, чтобы получать сгенерированные текстовые ответы.

Особенности

- Проект ориентирован на обработку и генерацию текстов на **русском языке**.
- Графический интерфейс поддерживает кириллицу и может быть адаптирован под нужды пользователя.
- Возможна настройка структуры папок и путей.


Требования

Убедитесь, что все зависимости установлены. Используйте файл `requirements.txt`:


Automated Text Processing and Generation System Based on VAE and Q-Networks

 Switching Between CNN and LSTM VAE

By default, the **CNN-VAE** model is used (`vae.py`).  
To switch to the **LSTM-VAE** model (`vae1.py`), follow these steps:

 Using LSTM-VAE:
1. Rename `vae.py` → `vae_backup.py`
2. Rename `vae1.py` → `vae.py`

 Reverting to CNN-VAE:
1. Rename `vae_backup.py` → `vae.py`
2. Rename current `vae.py` (LSTM version) → `vae1.py`
Warning**: Always ensure file safety during renaming. Using version control (e.g., Git) is highly recommended. Project Structure

- `main_menu.py` — Graphical User Interface (GUI)
- `vae.py` — CNN-based Variational Autoencoder (default)
- `vae1.py` — LSTM-based Variational Autoencoder (alternative)
- `q.py` — Q-network implementation
- `lem1.py` — Text lemmatization module
- `morph_singleton.py` — Wrapper for `pymorphy2` (morphological analyzer for Russian language)
 Usage

1. Place your text files into the directory:  
   `папка - текст/название_темы*/`
2. Launch the application:  
   ```bash
   python main_menu.py
   ```
3. Select a training mode:
**Auto-train** — fully automated training
**Step-by-step training** — manual control over each stage
4. Go to the **"Generation"** tab to generate responses after training.

Features

- Designed for processing and generating **Russian-language texts**.
- GUI supports Cyrillic and can be customized.
- Folder structure and UI paths are flexible.

