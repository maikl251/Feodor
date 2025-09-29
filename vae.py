import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, RepeatVector, TimeDistributed, Embedding, Layer, Dropout, Conv1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import pickle
import os
import logging
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import chardet
import re
import datetime

try:
    from pymorphy2 import MorphAnalyzer
    from morph_singleton import get_morph
except ImportError:
    print("pymorphy2 не установлен. Используется упрощённая лемматизация.")
    MorphAnalyzer = None
    def get_morph():
        return None

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        return super().get_config()

class KLAnealing(tf.keras.callbacks.Callback):
    def __init__(self, beta_start=0.0, beta_end=1.0, steps=10000):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps
        self.current_step = 0
    
    def on_batch_end(self, batch, logs=None):
        self.current_step += 1
        beta = min(self.beta_start + (self.beta_end - self.beta_start) * (self.current_step / self.steps), self.beta_end)
        K.set_value(self.model.beta, beta)
        logging.info(f"KL-annealing beta: {beta:.4f}")

class ImprovedVAELossLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = K.variable(0.0)
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        
        # Reconstruction loss
        xent_loss = tf.keras.losses.sparse_categorical_crossentropy(x, x_decoded_mean, from_logits=True)
        xent_loss = tf.reduce_mean(xent_loss)
        
        # KL divergence with free bits
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.maximum(kl_loss, 0.1)  # Free bits trick
        
        total_loss = xent_loss + self.beta * kl_loss
        self.add_loss(total_loss)
        return x_decoded_mean
    
    def compute_output_shape(self, input_shape):
        return input_shape[1]  # Форма x_decoded_mean
    
    def get_config(self):
        return super().get_config()

class GenerationMonitor(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, decoder, latent_dim, vocab_size, max_sequence_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
    
    def on_epoch_end(self, epoch, logs=None):
        z_sample = np.random.normal(size=(1, self.latent_dim))
        generated = self.decoder.predict(z_sample, verbose=0)[0]
        probs = generated / 1.0  # Temperature=1.0 for monitoring
        probs = np.exp(probs) / np.sum(np.exp(probs), axis=-1, keepdims=True)
        
        generated_indices = []
        for t in range(self.max_sequence_length):
            word_idx = np.random.choice(self.vocab_size, p=probs[t])
            if word_idx == 0:
                break
            generated_indices.append(word_idx)
        
        words = [self.tokenizer.index_word.get(idx, '<OOV>') for idx in generated_indices]
        generated_text = ' '.join(words)
        logging.info(f"Generated sample at epoch {epoch}: {generated_text}")

class TextVAE:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.model = None
        self.encoder = None
        self.decoder = None
        self.tokenizer = None
        self.max_sequence_length = 200  # Увеличено для динамического определения
        self.vocab_size = 10000
        self.latent_dim = 256  # Увеличено для лучшего качества
        self.intermediate_dim = 512
        self.is_trained = False
        self.corpus_base_file = "base_corpus.json"
        self.processed_files_file = "processed_files.json"
        self.qa_pairs_file = "qa_pairs.json"
        self.model_dir = "vae_model"
        self.training_data = []
        self.themes_data = {}
        self.processed_files = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.initial_epoch = 0
        self.loss_threshold = 0.01
        self.full_corpus = []
        self.model_compiled = False
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/app.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def load_processed_files(self):
        if os.path.exists(self.processed_files_file):
            try:
                with open(self.processed_files_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_processed_files(self, processed_files):
        unique_files = list(set(processed_files))
        with open(self.processed_files_file, 'w', encoding='utf-8') as f:
            json.dump(unique_files, f, ensure_ascii=False, indent=2)
    
    def load_corpus_base(self):
        if os.path.exists(self.corpus_base_file):
            try:
                with open(self.corpus_base_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_to_corpus_base(self, lemmatized_texts):
        base_data = self.load_corpus_base()
        base_data.extend(lemmatized_texts)
        unique_texts = list(set([text for text in base_data if text.strip()]))
        with open(self.corpus_base_file, 'w', encoding='utf-8') as f:
            json.dump(unique_texts, f, ensure_ascii=False, indent=2)
        return unique_texts
    
    def load_themes_data(self):
        themes_file = "themes_data.json"
        if os.path.exists(themes_file):
            try:
                with open(themes_file, 'r', encoding='utf-8') as f:
                    self.themes_data = json.load(f)
            except:
                self.themes_data = {}
        else:
            self.themes_data = {}
    
    def save_themes_data(self):
        with open("themes_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.themes_data, f, ensure_ascii=False, indent=2)
    
    def scan_for_new_corpora(self):
        self.processed_files = self.load_processed_files()
        texts_dir = "тексты"
        new_files_found = False
        new_texts = []
        self.load_themes_data()
        themes_with_texts = self.themes_data.copy()
        
        morph = get_morph()
        
        for file_path in glob.glob(os.path.join(texts_dir, "**", "*.txt"), recursive=True):
            file_key = file_path
            if file_key not in self.processed_files:
                try:
                    with open(file_path, 'rb') as file:
                        raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] if result['encoding'] and result['confidence'] > 0.7 else 'utf-8'
                    
                    try:
                        text_content = raw_data.decode(encoding)
                    except:
                        text_content = raw_data.decode('utf-8', errors='ignore')
                    
                    words = re.findall(r'\b\w+\b', text_content.lower())
                    if morph:
                        lemmatized_text = ' '.join(morph.parse(word)[0].normal_form for word in words)
                    else:
                        lemmatized_text = ' '.join(words)
                    
                    if lemmatized_text.strip():
                        new_texts.append(lemmatized_text)
                        theme = os.path.basename(os.path.dirname(file_path))
                        if theme not in themes_with_texts:
                            themes_with_texts[theme] = []
                        themes_with_texts[theme].append(lemmatized_text)
                        self.processed_files.append(file_key)
                        new_files_found = True
                        
                except Exception as e:
                    logging.error(f"Ошибка чтения {file_path}: {e}")
        
        if new_files_found:
            self.processed_files = list(set(self.processed_files))
            self.save_processed_files(self.processed_files)
            self.full_corpus = self.save_to_corpus_base(new_texts)
            self.themes_data = themes_with_texts
            self.save_themes_data()
            
            themes_info = {
                "total_themes": len(themes_with_texts),
                "themes": list(themes_with_texts.keys()),
                "texts_per_theme": {k: len(v) for k, v in themes_with_texts.items()},
                "last_update": datetime.datetime.now().isoformat()
            }
            with open("themes_info.json", 'w', encoding='utf-8') as f:
                json.dump(themes_info, f, ensure_ascii=False, indent=2)
            logging.info(f"[vae] Scan: Добавлено {len(new_texts)} текстов, full_corpus: {len(self.full_corpus)}")
        else:
            self.full_corpus = self.load_corpus_base()
            
        return new_texts
    
    def residual_block(self, x, filters, kernel_size=5):
        input_channels = x.shape[-1]
        
        shortcut = x
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        x = Dropout(0.3)(x)
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        
        if input_channels != filters:
            shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        
        return tf.keras.layers.add([x, shortcut])
    
    def build_model(self):
        inputs = Input(shape=(self.max_sequence_length,))
        x = Embedding(self.vocab_size, 512, mask_zero=True)(inputs) # Увеличиваем embedding dim 128->256->512 критически важно
        x = Conv1D(64, 5, activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 128)
        x = GlobalMaxPooling1D()(x)
        
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(self.max_sequence_length * 64)(latent_inputs)
        x = Reshape((self.max_sequence_length, 64))(x)
        
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 128)
        outputs = TimeDistributed(Dense(self.vocab_size))(x)
        
        self.decoder = Model(latent_inputs, outputs, name="decoder")
        
        vae_loss_layer = ImprovedVAELossLayer()
        
        reconstructed = self.decoder(z)
        vae_outputs = vae_loss_layer([inputs, reconstructed, z_mean, z_log_var])
        
        self.model = Model(inputs, vae_outputs, name="vae")
        self.model.beta = K.variable(0.0)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )
        
        self.model.compile(optimizer=optimizer)
        self.model_compiled = True
        logging.info("Модель VAE построена и скомпилирована.")
    
    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        try:
            self.encoder.save(os.path.join(model_dir, "encoder.keras"))
            self.decoder.save(os.path.join(model_dir, "decoder.keras"))
            self.model.save(os.path.join(model_dir, "vae_model.keras"))
            with open(os.path.join(model_dir, "tokenizer.pkl"), 'wb') as f:
                pickle.dump(self.tokenizer, f)
            logging.info("Модель и токенизатор сохранены.")
        except Exception as e:
            logging.error(f"Ошибка сохранения модели: {e}")
    
    def load_model(self, model_dir):
        try:
            self.encoder = load_model(os.path.join(model_dir, "encoder.keras"), custom_objects={'Sampling': Sampling})
            self.decoder = load_model(os.path.join(model_dir, "decoder.keras"))
            self.model = load_model(os.path.join(model_dir, "vae_model.keras"), custom_objects={'Sampling': Sampling, 'ImprovedVAELossLayer': ImprovedVAELossLayer})
            with open(os.path.join(model_dir, "tokenizer.pkl"), 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.model_compiled = True
            self.is_trained = True
            logging.info("Модель и токенизатор загружены.")
            return True
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def prepare_training_data(self, texts):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(
                num_words=self.vocab_size, 
                oov_token='<OOV>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
        
        # Фильтрация слишком коротких текстов
        filtered_texts = [text for text in texts if len(text.split()) > 5]
        
        self.tokenizer.fit_on_texts(filtered_texts)
        sequences = self.tokenizer.texts_to_sequences(filtered_texts)
        
        # Динамическое определение max_sequence_length
        seq_lengths = [len(seq) for seq in sequences]
        self.max_sequence_length = min(int(np.percentile(seq_lengths, 95)), 200)
        
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        augmented_sequences = []
        for seq in padded_sequences:
            if random.random() < 0.1:
                mask = np.random.choice([0, 1], size=len(seq), p=[0.1, 0.9])
                seq = np.where(mask == 0, 0, seq)
            augmented_sequences.append(seq)
        
        return np.array(augmented_sequences)
    
    def train_model(self, epochs=10, batch_size=32, new_texts=None, use_early_stopping=True, loss_threshold=0.01):
        self.loss_threshold = loss_threshold
        if new_texts:
            logging.info(f"[vae] Режим дообучения. Текстов в корпусе: {len(self.full_corpus)}")
            logging.info(f"[vae] Обнаружено новых текстов: {len(new_texts)}. Обновляю токенизатор.")
            self.full_corpus.extend(new_texts)
            self.full_corpus = list(set([text for text in self.full_corpus if text.strip()]))
            self.training_data = self.prepare_training_data(self.full_corpus)
            logging.info(f"[vae] Токенизатор обновлен. Новый размер словаря: {len(self.tokenizer.word_index)}")
        else:
            self.full_corpus = self.load_corpus_base()
            if not self.full_corpus:
                logging.warning("[vae] Корпус пуст. Обучение невозможно.")
                return None
            self.training_data = self.prepare_training_data(self.full_corpus)
        
        if self.model is None:
            if not self.load_model(self.model_dir):
                logging.info("[vae] Инициализация новой модели...")
                self.build_model()
        
        max_epochs = min(epochs, 200)
        actual_batch_size = min(batch_size, max(8, len(self.training_data) // 2))
        
        logging.info(f"[vae] Параметры обучения: эпохи={max_epochs}, batch_size={actual_batch_size}")
        
        try:
            callbacks = [KLAnealing()]
            callbacks.append(GenerationMonitor(
                self.tokenizer, 
                self.decoder, 
                self.latent_dim, 
                self.vocab_size, 
                self.max_sequence_length
            ))
            if use_early_stopping:
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=loss_threshold
                )
                reduce_lr = ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
                callbacks.extend([early_stopping, reduce_lr])
            
            history = self.model.fit(
                self.training_data,
                epochs=max_epochs,
                batch_size=actual_batch_size,
                validation_split=0.15 if len(self.training_data) > 20 else 0.0,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=self.initial_epoch
            )
            
            self.is_trained = True
            self.initial_epoch += len(history.history['loss'])
            self.save_model(self.model_dir)
            
            logging.info("[vae] Обучение завершено")
            if hasattr(history, 'history') and 'loss' in history.history:
                logging.info(f"[vae] Финальные потери: {history.history['loss'][-1]:.4f}")
                logging.info(f"[vae] Количество эпох: {len(history.history['loss'])}")
            
            return history
        except Exception as e:
            logging.error(f"[vae] Ошибка обучения: {e}")
            return None
    
    def get_topic_centroids(self, texts, themes):
        try:
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
            z_mean, _, _ = self.encoder.predict(padded, verbose=0)
            centroids = {}
            for theme in set(themes):
                theme_indices = [i for i, t in enumerate(themes) if t == theme]
                if theme_indices:
                    theme_z_mean = z_mean[theme_indices]
                    centroids[theme] = np.mean(theme_z_mean, axis=0)
            return centroids
        except Exception as e:
            logging.error(f"Ошибка вычисления центроидов: {e}")
            return {}
    
    def generate_queries_from_centroids(self, centroids, num_queries_per_topic, temperature):
        generated_pairs = []
        try:
            for theme, centroid in centroids.items():
                for _ in range(num_queries_per_topic):
                    z = np.random.normal(loc=centroid, scale=temperature, size=(1, self.latent_dim))
                    sequence = self.decoder.predict(z, verbose=0)[0]
                    probs = sequence / temperature
                    probs = np.exp(probs) / np.sum(np.exp(probs), axis=-1, keepdims=True)
                    
                    generated_indices = []
                    for t in range(self.max_sequence_length):
                        word_idx = np.random.choice(self.vocab_size, p=probs[t])
                        if word_idx == 0:
                            break
                        generated_indices.append(word_idx)
                    
                    words = [self.tokenizer.index_word.get(idx, '<OOV>') for idx in generated_indices]
                    text = ' '.join(words)
                    mid = len(words) // 2
                    question = ' '.join(words[:mid])
                    answer = ' '.join(words[mid:])
                    if question.strip() and answer.strip():
                        generated_pairs.append({'question': question, 'answer': answer, 'theme': theme})
        except Exception as e:
            logging.error(f"Ошибка генерации запросов: {e}")
        return generated_pairs
    
    def save_generated_pairs(self, generated_pairs):
        try:
            existing_pairs = []
            if os.path.exists(self.qa_pairs_file):
                with open(self.qa_pairs_file, 'r', encoding='utf-8') as f:
                    existing_pairs = json.load(f)
            
            existing_pairs.extend(generated_pairs)
            with open(self.qa_pairs_file, 'w', encoding='utf-8') as f:
                json.dump(existing_pairs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Ошибка сохранения пар: {e}")
    
    def generate_queries(self, num_queries_per_topic=5, temperature=0.8):
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        if not self.themes_data:
            raise ValueError("Нет данных о темах")
        
        all_texts = []
        all_themes = []
        for theme, texts in self.themes_data.items():
            all_texts.extend(texts)
            all_themes.extend([theme] * len(texts))
        
        if not all_texts:
            return 0
            
        centroids = self.get_topic_centroids(all_texts, all_themes)
        generated_pairs = self.generate_queries_from_centroids(centroids, num_queries_per_topic, temperature)
        self.save_generated_pairs(generated_pairs)
        return len(generated_pairs)



