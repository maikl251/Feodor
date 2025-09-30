Автоматизированная система обработки и генерации текстов на основе VAE и Q-сетей
Переключение между архитектурами VAE и Q-сети
Система поддерживает несколько архитектур нейросетей:
VAE модели:
- CNN-VAE (`vae.py`) - используется по умолчанию
- LSTM-VAE (`vae1.py`) - альтернативная архитектура
Q-сети:
- LSTM Q-network (`q.py`) - используется по умолчанию
- Transformer Q-network (`q1.py`) - альтернативная архитектура
  
➤ Активация LSTM-VAE + Transformer Q-network:
1. Переименуйте `vae.py` → `vae_backup.py` и `q.py` → `q_backup.py`
2. Переименуйте `vae1.py` → `vae.py` и `q1.py` → `q.py`
   
➤ Возврат к CNN-VAE + LSTM Q-network:
1. Переименуйте `vae_backup.py` → `vae.py` и `q_backup.py` → `q.py`
2. Переименуйте текущие `vae.py` → `vae1.py` и `q.py` → `q1.py`
   
➤ Смешанные конфигурации:
- CNN-VAE + Transformer**: `vae.py` + `q1.py` → `q.py`
- LSTM-VAE + LSTM**: `vae1.py` → `vae.py` + `q.py`
  
> Важно: Всегда создавайте резервные копии файлов. Рекомендуется использовать Git для контроля версий.

Структура проекта

- `main_menu.py` — графический интерфейс (GUI)
- `vae.py` — модель вариационного автоэнкодера на основе CNN (по умолчанию)
- `vae1.py` — модель вариационного автоэнкодера на основе LSTM (альтернатива)
- `q.py` — реализация Q-сети LSTM (по умолчанию)
- `q1.py` — Q-сеть на основе Transformer (альтернатива)
- `lem1.py` — модуль лемматизации текста
- `morph_singleton.py` — обёртка для `pymorphy2` (морфологический анализатор для русского языка)

Использование

1. Поместите текстовые файлы в папку:  
   `папка - текст/название_темы/`
2. Запустите приложение командой:  
   ```bash
   python main_menu.py
   ```
3. Выберите режим работы:
   - Автообучение — автоматическое обучение без участия пользователя
   - Обучение по шагам — поэтапное управление процессом обучения
4. После завершения обучения перейдите на вкладку "Генерация", чтобы получать сгенерированные текстовые ответы.

Особенности

- Проект ориентирован на обработку и генерацию текстов на **русском языке**.
- Графический интерфейс поддерживает кириллицу и может быть адаптирован под нужды пользователя.
- Возможна настройка структуры папок и путей.


Требования

Убедитесь, что все зависимости установлены. Используйте файл `requirements.txt`:


Automated Text Processing and Generation System Based on VAE and Q-Networks
Switching Between VAE and Q-network Architectures
The system supports multiple neural network architectures:
VAE Models:
- CNN-VAE (`vae.py`) - used by default
- LSTM-VAE (`vae1.py`) - alternative architecture
Q-networks:
- LSTM Q-network (`q.py`) - used by default
- Transformer Q-network (`q1.py`) - alternative architecture
  
➤ Activating LSTM-VAE + Transformer Q-network:
1. Rename `vae.py` → `vae_backup.py` and `q.py` → `q_backup.py`
2. Rename `vae1.py` → `vae.py` and `q1.py` → `q.py`
   
➤ Reverting to CNN-VAE + LSTM Q-network:
1. Rename `vae_backup.py` → `vae.py` and `q_backup.py` → `q.py`
2. Rename current `vae.py` → `vae1.py` and `q.py` → `q1.py`
   
➤ Mixed Configurations:
- CNN-VAE + Transformer: `vae.py` + rename `q1.py` → `q.py`
- LSTM-VAE + LSTM**: rename `vae1.py` → `vae.py` + `q.py`
  
> Warning: Always create file backups. Using Git for version control is highly recommended.

 

- `main_menu.py` — Graphical User Interface (GUI)
- `vae.py` — CNN-based Variational Autoencoder (default)
- `vae1.py` — LSTM-based Variational Autoencoder (alternative)
- `q.py` — LSTM-based Q-network (default)
- `q1.py` — Transformer-based Q-network (alternative)
- `lem1.py` — Text lemmatization module
- `morph_singleton.py` — Wrapper for `pymorphy2` (morphological analyzer for Russian language)
 Usage

1. Place your text files into the directory:  
   `папка - текст/название_темы/`
2. Launch the application:  
   ```bash
   python main_menu.py
   ```
3. Select a training mode:
*Auto-train* — fully automated training
*Step-by-step training* — manual control over each stage
4. Go to the *"Generation"* tab to generate responses after training.

Features

- Designed for processing and generating *Russian-language texts*.
- GUI supports Cyrillic and can be customized.
- Folder structure and UI paths are flexible.

**Title:** Automated Text Processing and Generation System Based on VAE and Q-Networks  
**DOI:** [`10.5281/zenodo.17180095`](https://doi.org/10.5281/zenodo.17180095)  
**Archive:** Zenodo (CERN)
**Title:** Automated Text Processing and Generation System Based on VAE and Q-Networks (Part2).
**DOI:**  https://doi.org/10.5281/zenodo.17229330
**Archive:** Zenodo (CERN)





