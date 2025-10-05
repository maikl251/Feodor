import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, Lambda, Concatenate, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os
from collections import Counter, defaultdict
import re
import datetime
import logging
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import random

class MultiHeadAttentionWithHeads(tf.keras.layers.Layer):
    """Многоголовое внимание с явным разделением голов для разных аспектов"""
    def __init__(self, num_heads, key_dim, head_names, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_names = head_names
        self.attention_layers = {}
        
        for head_name in head_names:
            self.attention_layers[head_name] = tf.keras.layers.MultiHeadAttention(
                num_heads=1, key_dim=key_dim, name=f"{head_name}_attention"
            )
    
    def call(self, inputs, mask=None):
        attention_outputs = []
        
        for head_name in self.head_names:
            attn_output = self.attention_layers[head_name](
                inputs, inputs, attention_mask=mask
            )
            attention_outputs.append(attn_output)
        
        # Конкатенация всех голов
        return tf.concat(attention_outputs, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'head_names': self.head_names
        })
        return config

class QTextNetwork:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = 150
        self.vocab_size = 5000
        self.latent_dim = 64
        self.corpus_stats = defaultdict(int)
        self.corpus_bigrams = defaultdict(int)
        self.corpus_trigrams = defaultdict(int)
        self.corpus_sequences = []
        self.is_trained = False
        self.corpus_data = []
        self.query_answer_pairs = []
        self.keyword_chains = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, min_df=1)
        self.chain_vectors = np.array([]).reshape(0, 1000)
        self.corpus_base_file = "base_corpus.json"
        self.qa_pairs_file = "qa_pairs.json"
        self.model_dir = "q_learning_model"
        self.initial_epoch = 0
        self.loss_threshold = 0.01
        self.chain_cache = {}
        self.cache_size = 100
        self.model_compiled = False
        self.generation_cache = {}
        
        # СПЕЦИАЛИЗИРОВАННЫЕ ГОЛОВЫ БЕЗ CO-OCCURRENCE
        self.attention_head_names = ['semantic', 'syntax', 'ngram_pattern']
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/app.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.qa_pairs_list = []

    def load_all_query_answer_pairs(self):
        logging.info("Загрузка пар вопрос-ответ...")
        if os.path.exists(self.qa_pairs_file):
            try:
                with open(self.qa_pairs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Ошибка загрузки qa_pairs.json: {e}")
                return []
        return []
    
    def load_corpus_base(self):
        if os.path.exists(self.corpus_base_file):
            try:
                with open(self.corpus_base_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Ошибка загрузки корпуса: {e}")
                return []
        return []
    
    def auto_load_data(self):
        try:
            logging.info("Автозагрузка данных...")
            
            self.corpus_data = self.load_corpus_base()
            logging.info(f"Загружено текстов корпуса: {len(self.corpus_data)}")
            
            self.query_answer_pairs = self.load_all_query_answer_pairs()
            self.qa_pairs_list = self.query_answer_pairs
            logging.info(f"Загружено пар вопрос-ответ: {len(self.query_answer_pairs)}")
            
            if len(self.query_answer_pairs) == 0:
                logging.warning("Нет пар вопрос-ответ для обучения")
                return False
                
            self._build_corpus_chains()
            self._extract_keyword_chains()
            self._build_chain_vectors()
            logging.info(f"Сформировано цепочек ключевых слов: {len(self.keyword_chains)}")
            
            return True
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def _build_corpus_chains(self):
        self.corpus_sequences = []
        self.corpus_stats = defaultdict(int)
        self.corpus_bigrams = defaultdict(int)
        self.corpus_trigrams = defaultdict(int)
        
        logging.info("Формирование цепочек корпуса...")
        
        for text in self.corpus_data:
            if isinstance(text, str) and text.strip():
                words = text.split()
                if len(words) > 1:
                    self.corpus_sequences.append(words)
                    for word in words:
                        self.corpus_stats[word] += 1
                    
                    # Биграммы
                    for i in range(len(words) - 1):
                        bigram = ' '.join(words[i:i+2])
                        self.corpus_bigrams[bigram] += 1
                    
                    # Триграммы
                    for i in range(len(words) - 2):
                        trigram = ' '.join(words[i:i+3])
                        self.corpus_trigrams[trigram] += 1
        
        logging.info(f"Сформировано последовательностей корпуса: {len(self.corpus_sequences)}")
        logging.info(f"Уникальных слов: {len(self.corpus_stats)}")
        logging.info(f"Уникальных биграмм: {len(self.corpus_bigrams)}")
        logging.info(f"Уникальных триграмм: {len(self.corpus_trigrams)}")
    
    def _extract_keyword_chains(self):
        logging.info("Извлечение цепочек ключевых слов...")
        self.keyword_chains = []
        for pair in self.query_answer_pairs:
            question_words = pair['question'].split()
            answer_words = pair['answer'].split()
            chain = question_words + answer_words
            if len(chain) >= 2:
                self.keyword_chains.append({
                    'query': question_words,
                    'answer': answer_words,
                    'full_text': ' '.join(chain)
                })
    
    def _build_chain_vectors(self):
        logging.info("Создание векторов цепочек...")
        chain_texts = [chain['full_text'] for chain in self.keyword_chains]
        if chain_texts:
            try:
                self.tfidf_vectorizer.fit(chain_texts + self.corpus_data[:5])
                self.chain_vectors = self.tfidf_vectorizer.transform(chain_texts).toarray()
            except Exception as e:
                logging.error(f"Ошибка создания векторов цепочек: {e}")
                self.chain_vectors = np.array([]).reshape(0, 1000)
        else:
            if self.corpus_data:
                self.tfidf_vectorizer.fit(self.corpus_data[:10])
            else:
                self.tfidf_vectorizer.fit(['dummy text'])
            self.chain_vectors = np.array([]).reshape(0, 1000)

    def find_sentences_with_words(self, words):
        """Находит предложения в корпусе, содержащие указанные слова"""
        matching_sentences = []
        words_lower = [word.lower() for word in words]
        
        for text in self.corpus_data:
            if isinstance(text, str) and text.strip():
                text_lower = text.lower()
                # Проверяем, содержит ли текст хотя бы одно из искомых слов
                if any(word in text_lower for word in words_lower):
                    matching_sentences.append(text)
        
        logging.info(f"Найдено предложений с словами {words}: {len(matching_sentences)}")
        return matching_sentences

    def extract_semantic_core(self, text, min_common_words=1):
        """Извлечение семантического ядра из текста."""
        words = text.lower().split()
        word_counts = Counter(words)
        semantic_core = [word for word, count in word_counts.items() if count >= min_common_words]
        return semantic_core

    def word_in_pseudo_answers(self, word):
        """Проверка наличия слова в псевдоответах."""
        for pair in self.query_answer_pairs:
            if word.lower() in pair['answer'].lower().split():
                return True
        return False

    def word_pair_in_pseudo_answers(self, word_pair):
        """Проверка наличия пары слов в псевдоответах."""
        pair_text = ' '.join(word_pair).lower()
        for pair in self.query_answer_pairs:
            if pair_text in pair['answer'].lower():
                return True
        return False

    def filter_relevant_words(self, words, context):
        """Фильтрация релевантных слов на основе контекста."""
        context_words = context.lower().split()
        return [word for word in words if self.is_semantically_related(word, context_words)]

    def is_semantically_related(self, word, context_words):
        """Проверка семантической связи слова с контекстом."""
        word = word.lower()
        if word in context_words or self.word_in_pseudo_answers(word):
            return True
        for context_word in context_words:
            if self.word_pair_in_pseudo_answers([word, context_word]):
                return True
        return False

    def find_best_matching_answer(self, generated_text):
        """Находит наиболее подходящий ответ из qa_pairs для сгенерированного текста."""
        best_similarity = 0
        best_answer = ""
        
        for pair in self.query_answer_pairs:
            answer_text = pair['answer'].lower()
            generated_words = set(generated_text.lower().split())
            answer_words = set(answer_text.split())
            
            # Вычисляем overlap
            overlap = len(generated_words.intersection(answer_words))
            total_unique = len(generated_words.union(answer_words))
            
            if total_unique > 0:
                similarity = overlap / total_unique
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_answer = pair['answer']
        
        return best_answer, best_similarity

    def clean_generated_text(self, generated_text, reference_answer):
        """Очищает сгенерированный текст, оставляя только слова из reference_answer."""
        if not reference_answer:
            return generated_text
            
        ref_words = set(reference_answer.lower().split())
        generated_words = generated_text.split()
        
        # Оставляем только слова, которые есть в эталонном ответе
        cleaned_words = [word for word in generated_words if word.lower() in ref_words]
        
        # Если очистка удалила все слова, возвращаем оригинал
        if not cleaned_words:
            return generated_text
            
        return ' '.join(cleaned_words)

    def generate_multiple_variants(self, seed_words, num_variants=3, max_length=30):
        """Генерирует несколько вариантов текста из seed words."""
        variants = []
        
        for i in range(num_variants):
            # Немного меняем температуру для разнообразия
            temperature = 0.7 + (i * 0.1)
            variant, confidence = self.generate_text_from_seed(
                seed_words, 
                max_length=max_length, 
                temperature=temperature
            )
            variants.append((variant, confidence))
        
        return variants

    def select_best_variant(self, variants):
        """Выбирает лучший вариант на основе сравнения с ответами из qa_pairs."""
        best_variant = ""
        best_similarity = 0
        best_confidence = 0
        
        for variant, confidence in variants:
            # Находим наиболее подходящий ответ для этого варианта
            best_answer, similarity = self.find_best_matching_answer(variant)
            
            # Комбинируем similarity и confidence
            combined_score = (similarity + confidence) / 2
            
            if combined_score > best_similarity:
                best_similarity = combined_score
                best_variant = variant
                best_confidence = confidence
        
        # Очищаем лучший вариант
        best_answer, _ = self.find_best_matching_answer(best_variant)
        cleaned_variant = self.clean_generated_text(best_variant, best_answer)
        
        return cleaned_variant, best_confidence

    def get_enhanced_chain_reward(self, pseudo_question, pseudo_answer, generated_chain):
        """УПРОЩЕННАЯ функция вознаграждения БЕЗ CO-OCCURRENCE"""
        
        question_words = pseudo_question.split()
        answer_words = pseudo_answer.split()
        generated_words = generated_chain.split()
        
        # 1. БАЗОВОЕ СОВПАДЕНИЕ СЛОВ
        question_word_match = len(set(question_words).intersection(generated_words)) / max(len(question_words), 1)
        answer_word_match = len(set(answer_words).intersection(generated_words)) / max(len(answer_words), 1)
        
        # 2. СОВПАДЕНИЕ БИГРАММ
        question_bigrams = [' '.join(question_words[i:i+2]) for i in range(len(question_words)-1)]
        answer_bigrams = [' '.join(answer_words[i:i+2]) for i in range(len(answer_words)-1)]
        generated_bigrams = [' '.join(generated_words[i:i+2]) for i in range(len(generated_words)-1)]
        
        bigram_question_match = len(set(question_bigrams).intersection(generated_bigrams)) / max(len(question_bigrams), 1)
        bigram_answer_match = len(set(answer_bigrams).intersection(generated_bigrams)) / max(len(answer_bigrams), 1)
        
        # 3. СОВПАДЕНИЕ ТРИГРАММ
        question_trigrams = [' '.join(question_words[i:i+3]) for i in range(len(question_words)-2)]
        answer_trigrams = [' '.join(answer_words[i:i+3]) for i in range(len(answer_words)-2)]
        generated_trigrams = [' '.join(generated_words[i:i+3]) for i in range(len(generated_words)-2)]
        
        trigram_question_match = len(set(question_trigrams).intersection(generated_trigrams)) / max(len(question_trigrams), 1)
        trigram_answer_match = len(set(answer_trigrams).intersection(generated_trigrams)) / max(len(answer_trigrams), 1)
        
        # 4. ШТРАФ ЗА ПОВТОРЕНИЯ
        word_counts = Counter(generated_words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_penalty = repeated_words / len(generated_words) if generated_words else 0
        
        # 5. КОМБИНИРОВАННОЕ ВОЗНАГРАЖДЕНИЕ (БЕЗ CO-OCCURRENCE)
        total_reward = (
            0.25 * question_word_match +      # Слова вопроса
            0.25 * answer_word_match +        # Слова ответа
            0.15 * bigram_question_match +    # Биграммы вопроса
            0.15 * bigram_answer_match +      # Биграммы ответа
            0.10 * trigram_question_match +   # Триграммы вопроса
            0.10 * trigram_answer_match       # Триграммы ответа
        )
        
        # Применение штрафа за повторения
        total_reward *= (1 - repetition_penalty * 0.3)
        
        return min(total_reward, 1.0)

    def build_model(self):
        logging.info("Строительство улучшенной модели Q-сети с многоголовым трансформером...")
    
        inputs = Input(shape=(self.max_sequence_length,))
        x = Embedding(input_dim=self.vocab_size, output_dim=128, mask_zero=True)(inputs)
    
        # Позиционное кодирование
        positions = tf.range(start=0, limit=self.max_sequence_length, delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoding = tf.keras.layers.Embedding(
            input_dim=self.max_sequence_length, 
            output_dim=128
        )(positions)
        x = x + pos_encoding
    
        # Создание маски
        mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.not_equal(x, 0), tf.float32))(inputs)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
    
        # МНОГОГОЛОВЫЙ ТРАНСФОРМЕР СО СПЕЦИАЛИЗИРОВАННЫМИ ГОЛОВАМИ (БЕЗ CO-OCCURRENCE)
        attention_outputs = []
    
        for head_name in self.attention_head_names:
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=32, name=f"{head_name}_attention"
            )
            attn_output = attn_layer(x, x, attention_mask=mask)
            attention_outputs.append(attn_output)
    
        # Конкатенация выходов всех голов
        combined_attention = Concatenate()(attention_outputs)
    
        # Проекция для приведения размерности к исходной
        combined_attention_projected = Dense(128, name="attention_projection")(combined_attention)
    
        x = LayerNormalization(epsilon=1e-6)(x + combined_attention_projected)
    
        # Feed-forward сети для каждой головы
        ff_outputs = []
        for i, head_output in enumerate(attention_outputs):
            ff_out = Dense(256, activation='relu', name=f"ff_{i}_dense1")(head_output)
            ff_out = Dropout(0.1)(ff_out)
            ff_out = Dense(128, name=f"ff_{i}_dense2")(ff_out)
            ff_outputs.append(ff_out)
    
        combined_ff = Concatenate()(ff_outputs)
    
        # Проекция для приведения размерности к исходной
        combined_ff_projected = Dense(128, name="ff_projection")(combined_ff)
    
        x = LayerNormalization(epsilon=1e-6)(x + combined_ff_projected)
    
        # Выходной слой
        x = Dense(256, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x[:, -1, :])
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.vocab_size, activation='softmax')(x)
    
        self.model = Model(inputs, outputs)
        logging.info("Улучшенная модель построена успешно.")
        self.compile_model()

    def compile_model(self):
        if not self.model_compiled:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                clipnorm=1.0
            )
            loss_fn = 'categorical_crossentropy'
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy']
            )
            self.model_compiled = True
            logging.info("Модель скомпилирована с categorical_crossentropy loss.")
    
    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        try:
            self.model.save(os.path.join(model_dir, "q_model.keras"))
            with open(os.path.join(model_dir, "tokenizer.pkl"), 'wb') as f:
                pickle.dump(self.tokenizer, f)
                
            logging.info("Модель и токенизатор сохранены.")
        except Exception as e:
            logging.error(f"Ошибка сохранения модели: {e}")
    
    def load_from_dir(self, model_dir):
        try:
            self.model = load_model(os.path.join(model_dir, "q_model.keras"))
            with open(os.path.join(model_dir, "tokenizer.pkl"), 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            self.model_compiled = True
            self.is_trained = True
            logging.info("Модель и токенизатор загружены.")
            return True
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def load_model(self, model_dir):
        return self.load_from_dir(model_dir)
    
    def texts_to_sequences(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        return padded
    
    def train(self, epochs=5, batch_size=32, use_early_stopping=True, loss_threshold=0.01):
        self.generation_cache = {}
        self.loss_threshold = loss_threshold
        
        if not self.auto_load_data():
            logging.warning("Нет данных для обучения. Инициализация базовой модели...")
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(['текст запрос ответ данные обучение модель система'])
            self.build_model()
            self.is_trained = True
            return None
        
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
            all_texts = self.corpus_data + [pair['question'] + ' ' + pair['answer'] for pair in self.query_answer_pairs]
            self.tokenizer.fit_on_texts(all_texts)
        
        if self.model is None:
            self.build_model()
        
        state_sequences = []
        targets = []
        
        logging.info("Подготовка УЛУЧШЕННЫХ данных для обучения с правильными target векторами...")
        
        # ОБУЧЕНИЕ НА ПРЕДЛОЖЕНИЯХ ИЗ КОРПУСА ПО ЦЕНТРОИДАМ
        for pair in self.query_answer_pairs:
            # Извлекаем слова из вопроса (центроид A)
            centroid_words = pair['question'].split()
            
            # Ищем предложения в корпусе, содержащие эти слова
            matching_sentences = self.find_sentences_with_words(centroid_words)
            
            for sentence in matching_sentences:
                words = sentence.split()
                
                # Делим предложение на части для обучения склейке
                for i in range(1, min(len(words), 8)):
                    current_state = words[:i]
                    next_word = words[i]
                    
                    current_state_text = ' '.join(current_state)
                    state_sequence = self.texts_to_sequences([current_state_text])[0]
                    
                    if not np.any(state_sequence):
                        continue
                    
                    state_sequences.append(state_sequence)
                    
                    # ИСПРАВЛЕНИЕ: ПРАВИЛЬНЫЕ target векторы для categorical_crossentropy
                    target_vector = np.zeros(self.vocab_size, dtype=np.float32)
                    
                    if next_word in self.tokenizer.word_index:
                        word_idx = self.tokenizer.word_index[next_word]
                        if word_idx < self.vocab_size:
                            # Основное слово получает высокую вероятность
                            target_vector[word_idx] = 0.6  # Фиксированная высокая вероятность для корпусных данных
                    
                    # Распределяем оставшуюся вероятность
                    remaining_prob = 1.0 - np.sum(target_vector)
                    if remaining_prob > 0:
                        uniform_value = remaining_prob / self.vocab_size
                        target_vector += uniform_value
                    
                    # Финальная нормализация (сумма должна быть точно 1.0)
                    target_sum = np.sum(target_vector)
                    if target_sum > 0:
                        target_vector = target_vector / target_sum
                    else:
                        # Запасной вариант если все нули
                        target_vector = np.ones(self.vocab_size) / self.vocab_size
                    
                    targets.append(target_vector)
        
        # ДОПОЛНИТЕЛЬНОЕ ОБУЧЕНИЕ НА КОРПУСЕ
        logging.info("Дополнительное обучение на корпусе...")
        for text in self.corpus_data[:1000]:  # Ограничим для скорости
            words = text.split()
            for i in range(1, min(len(words), 6)):
                current_state = words[:i]
                next_word = words[i]
                
                current_state_text = ' '.join(current_state)
                state_sequence = self.texts_to_sequences([current_state_text])[0]
                
                if not np.any(state_sequence):
                    continue
                
                state_sequences.append(state_sequence)
                
                # Базовая награда для корпусных данных с правильным форматом
                target_vector = np.zeros(self.vocab_size, dtype=np.float32)
                
                if next_word in self.tokenizer.word_index:
                    word_idx = self.tokenizer.word_index[next_word]
                    if word_idx < self.vocab_size:
                        target_vector[word_idx] = 0.6  # Высокая вероятность для корпусных данных
                
                # Распределяем оставшуюся вероятность
                remaining_prob = 1.0 - np.sum(target_vector)
                if remaining_prob > 0:
                    uniform_value = remaining_prob / self.vocab_size
                    target_vector += uniform_value
                
                # Нормализация
                target_vector = target_vector / np.sum(target_vector)
                targets.append(target_vector)
        
        if len(state_sequences) < 10:
            logging.warning("Слишком мало данных для обучения. Добавление базовых примеров.")
            basic_words = ['текст', 'запрос', 'ответ', 'данные', 'обучение', 'модель', 'система']
            for i, word in enumerate(basic_words):
                if word in self.tokenizer.word_index:
                    word_idx = self.tokenizer.word_index[word]
                    if word_idx < self.vocab_size:
                        state_sequence = [word_idx]
                        state_sequence = pad_sequences([state_sequence], 
                                                     maxlen=self.max_sequence_length, 
                                                     padding='post')[0]
                        state_sequences.append(state_sequence)
                        
                        target_vector = np.zeros(self.vocab_size, dtype=np.float32)
                        next_word = basic_words[(i + 1) % len(basic_words)]
                        next_word_idx = self.tokenizer.word_index.get(next_word, 1)
                        if next_word_idx < self.vocab_size:
                            target_vector[next_word_idx] = 0.7
                        
                        # Распределение оставшейся вероятности
                        remaining_prob = 1.0 - np.sum(target_vector)
                        if remaining_prob > 0:
                            uniform_value = remaining_prob / self.vocab_size
                            target_vector += uniform_value
                        
                        target_vector = target_vector / np.sum(target_vector)
                        targets.append(target_vector)
        
        if len(state_sequences) < 5:
            logging.error("Недостаточно данных для обучения")
            return None
            
        state_sequences = np.array(state_sequences)
        targets = np.array(targets)
        
        # ПРОВЕРКА: убедимся что targets валидны
        target_sums = np.sum(targets, axis=1)
        logging.info(f"Проверка target векторов:")
        logging.info(f"  - Min sum: {np.min(target_sums):.6f}")
        logging.info(f"  - Max sum: {np.max(target_sums):.6f}")
        logging.info(f"  - Mean sum: {np.mean(target_sums):.6f}")
        
        logging.info(f"Сформировано обучающих последовательностей: {len(state_sequences)}")
        
        actual_epochs = min(epochs, 1000)
        actual_batch_size = min(batch_size, max(8, len(state_sequences) // 2))
        
        logging.info(f"Параметры обучения: эпохи={actual_epochs}, batch_size={actual_batch_size}")
        
        try:
            callbacks = []
            if use_early_stopping:
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=5,  # Увеличим patience
                    restore_best_weights=True,
                    min_delta=0.001
                )
                reduce_lr = ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
                callbacks = [early_stopping, reduce_lr]
            
            self.compile_model()
            
            history = self.model.fit(
                state_sequences,
                targets,
                epochs=actual_epochs,
                batch_size=actual_batch_size,
                validation_split=0.15 if len(state_sequences) > 20 else 0.0,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=self.initial_epoch
            )
            
            logging.info("Обучение улучшенной Q-сети завершено")
            self.is_trained = True
            
            if hasattr(history, 'history') and 'loss' in history.history:
                final_loss = history.history['loss'][-1]
                num_epochs_trained = len(history.history['loss'])
                self.initial_epoch += num_epochs_trained
                logging.info(f"Финальные потери: {final_loss:.4f}")
                logging.info(f"Количество обученных эпох: {num_epochs_trained}")
            else:
                logging.warning("Не удалось получить информацию о потерях из history")
            
            self.save_model(self.model_dir)
            return history
        except Exception as e:
            logging.error(f"Ошибка во время обучения: {e}")
            return None
    
    def reset_generation_state(self):
        """Сброс состояния генерации для нового запроса."""
        # Очищаем кеш генерации
        self.generation_cache = {}
        logging.info("Состояние генерации сброшено")

    def find_similar_question(self, query):
        try:
            query_vector = self.tfidf_vectorizer.transform([query]).toarray()[0]
            similarities = []
            for pair in self.query_answer_pairs:
                question = pair['question']
                question_vector = self.tfidf_vectorizer.transform([question]).toarray()[0]
                similarity = cosine_similarity([query_vector], [question_vector])[0][0]
                similarities.append((pair, similarity))
            if similarities:
                best_pair, similarity_score = max(similarities, key=lambda x: x[1])
                return best_pair, similarity_score
            return None, 0.0
        except:
            return None, 0.0

    def generate_reasoning_chain(self, initial_query, max_iterations=3, max_length_per_step=30):
        """Генерация цепочки рассуждений по новому алгоритму."""
        current_text = initial_query
        reasoning_chain = [current_text]  # Начинаем с исходного запроса
        confidence = 1.0
        
        for iteration in range(max_iterations):
            logging.info(f"Итерация {iteration + 1}: {current_text}")
            
            # 1. Извлечение семантического ядра
            semantic_core = self.extract_semantic_core(current_text, min_common_words=1)
            if not semantic_core:
                logging.info("Не удалось извлечь семантическое ядро")
                break
                
            # 2. Фильтрация релевантных слов
            filtered_words = self.filter_relevant_words(semantic_core, current_text)
            if not filtered_words:
                logging.info("Не осталось релевантных слов после фильтрации")
                break
            
            # 3. Генерация нескольких вариантов
            variants = self.generate_multiple_variants(
                filtered_words, 
                num_variants=3, 
                max_length=max_length_per_step
            )
            
            # 4. Выбор и очистка лучшего варианта
            best_variant, step_confidence = self.select_best_variant(variants)
            
            # 5. Проверка на стагнацию (если новый текст почти не отличается от предыдущего)
            if best_variant and best_variant != current_text:
                reasoning_chain.append(best_variant)
                confidence = min(confidence, step_confidence)
                current_text = best_variant
                
                # Условие остановки: достигнута максимальная длина или текст перестал меняться
                if len(best_variant.split()) >= max_length_per_step:
                    break
                    
                # Проверяем, не повторяется ли текст
                if len(reasoning_chain) >= 2 and best_variant == reasoning_chain[-2]:
                    break
            else:
                break
        
        return reasoning_chain, confidence

    
    
    def generate_text(self, query, max_length=30, temperature=1.0):
        """Основной метод генерации текста с итеративной цепочкой рассуждений."""
        try:
            # Используем итеративную генерацию цепочки
            reasoning_chain, confidence = self.generate_reasoning_chain(query, max_iterations=3)
            
            # Возвращаем ВСЮ цепочку как результат
            full_response = " → ".join(reasoning_chain)
            
            logging.info(f"Сгенерирована полная цепочка: {full_response}")
            logging.info(f"Уверенность: {confidence:.2f}")
            
            # Автоматический сброс после генерации
            self.reset_generation_state()
            
            return full_response, confidence
        except Exception as e:
            logging.error(f"Ошибка в generate_text: {e}")
            self.reset_generation_state()
            return query, 0.5
    
    def generate_text_from_seed(self, seed_sequence, max_length=20, temperature=1.0):
        try:
            # Проверка на пустой seed
            if not seed_sequence or len(seed_sequence) == 0:
                seed_sequence = ['текст', 'запрос']  # Запасной вариант
                
            seed_text = ' '.join(seed_sequence)
            sequence = self.texts_to_sequences([seed_text])[0]
            generated_words = seed_sequence[:]
            confidence = 1.0
            
            for step in range(max_length):
                cache_key = tuple(sequence)
                if cache_key in self.generation_cache:
                    logits = self.generation_cache[cache_key]
                else:
                    padded_sequence = pad_sequences([sequence], maxlen=self.max_sequence_length, padding='post')
                    logits = self.model.predict(padded_sequence, verbose=0)[0]
                    if len(self.generation_cache) < self.cache_size:
                        self.generation_cache[cache_key] = logits
                
                probs = tf.nn.softmax(logits / max(1e-8, temperature)).numpy()
                next_word_idx = np.random.choice(len(probs), p=probs)
                
                if next_word_idx == 0 or next_word_idx >= self.vocab_size:
                    break
                
                next_word = self.tokenizer.index_word.get(next_word_idx, '<OOV>')
                if next_word == '<OOV>':
                    break
                generated_words.append(next_word)
                sequence = np.append(sequence[1:], next_word_idx)
                
                # Обновление уверенности на основе упрощенной функции вознаграждения
                current_text = ' '.join(generated_words)
                if len(seed_sequence) > 0:
                    reward = self.get_enhanced_chain_reward(
                        ' '.join(seed_sequence[:len(seed_sequence)//2]),
                        ' '.join(seed_sequence[len(seed_sequence)//2:]),
                        current_text
                    )
                    confidence = min(confidence, reward)
                
                if next_word in ['.', '!', '?']:
                    break
            
            return ' '.join(generated_words), confidence
        except Exception as e:
            logging.error(f"Ошибка генерации текста: {e}")
            return ' '.join(seed_sequence), 0.5
