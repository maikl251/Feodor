import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Input, Dropout, Lambda, Concatenate, LayerNormalization, GlobalMaxPooling1D,GRU, Bidirectional 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os
from collections import Counter, defaultdict
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from tensorflow.keras.losses import CategoricalCrossentropy

def _pad_mask(x):
    return tf.cast(tf.not_equal(x, 0), tf.float32)

def _causal_mask(x):
    seq_len = tf.shape(x)[1]
    return tf.linalg.band_part(
        tf.ones((seq_len, seq_len), dtype=tf.float32),
        num_lower=-1, num_upper=0
    )[tf.newaxis, tf.newaxis, :, :]

def _combine_masks(masks):
    return tf.minimum(masks[0], masks[1])

def _square_features(t):
    return tf.concat([t, tf.square(t)], axis=-1)

class TopKCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, k=9000, window_ratio=0.01, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.window_ratio = window_ratio 

    def update_window_ratio(self, new_ratio):
        """Обновить размер окна"""
        self.window_ratio = min(1.0, max(0.01, new_ratio))

    def call(self, y_true, y_pred):
        # Стабилизация: обрезаем экстремально малые/большие значения
        y_pred = tf.clip_by_value(y_pred, 1e-12, 1.0 - 1e-12)
        batch_size = tf.shape(y_pred)[0]
        vocab_size = tf.shape(y_pred)[1]

        # Оригинальный расчет K
        effective_k = tf.cast(tf.cast(self.k, tf.float32) * self.window_ratio, tf.int32)
        k = tf.minimum(effective_k, vocab_size)
        k = tf.maximum(1, k) 

        topk_values, topk_indices = tf.math.top_k(y_pred, k=k)
        original_topk_indices = tf.identity(topk_indices)

        correct_indices = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        correct_expanded = tf.tile(tf.expand_dims(correct_indices, axis=1), [1, k])
        
        already_in_topk = tf.reduce_any(tf.equal(correct_expanded, topk_indices), axis=1)
        
        batch_range = tf.range(batch_size, dtype=tf.int32)
        mask = tf.logical_not(already_in_topk)
        batches_needing_update = tf.boolean_mask(batch_range, mask)
        num_batches_needing = tf.shape(batches_needing_update)[0]
        
        def update_topk():
            num_replace = tf.maximum(1, tf.cast(tf.cast(k, tf.float32) * 0.1, tf.int32))
            replace_positions = tf.random.uniform([num_batches_needing, num_replace], 0, k, dtype=tf.int32)
            batch_indices = tf.tile(tf.expand_dims(batches_needing_update, axis=1), [1, num_replace])
            update_indices = tf.reshape(tf.stack([batch_indices, replace_positions], axis=-1), [-1, 2])
            correct_for_update = tf.tile(tf.gather(correct_indices, batches_needing_update)[:, tf.newaxis], [1, num_replace])
            return tf.tensor_scatter_nd_update(topk_indices, update_indices, tf.reshape(correct_for_update, [-1]))
        
        topk_indices = tf.cond(tf.greater(num_batches_needing, 0), update_topk, lambda: topk_indices)

        # --- ИСПРАВЛЕНИЕ: МЫ ДОЛЖНЫ ОБНОВИТЬ VALUES ПОСЛЕ ОБНОВЛЕНИЯ INDICES ---
        batch_indices_tile = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, k])
        gather_indices = tf.stack([batch_indices_tile, topk_indices], axis=-1)
        
        # Берем значения из y_pred (уже вероятности!), соответствующие новым индексам
        actual_topk_values = tf.gather_nd(y_pred, gather_indices)
        
        # Если в y_pred уже вероятности, Softmax НЕ НУЖЕН. Используем значения напрямую.
        # Это предотвратит раздувание лосса до 23.
        topk_probs = actual_topk_values 

        y_true_topk = tf.gather_nd(y_true, gather_indices)
        has_target = tf.reduce_sum(y_true_topk, axis=-1) > 0

        epsilon = tf.constant(1e-9, dtype=y_pred.dtype)
        ce = -tf.reduce_sum(y_true_topk * tf.math.log(topk_probs + epsilon), axis=-1)

        # Твой оригинальный fallback
        fallback_loss = tf.math.log(tf.cast(vocab_size, dtype=y_pred.dtype))
        return tf.where(has_target, ce, fallback_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'k': self.k,
            'window_ratio': self.window_ratio  
        })
        return config


class QPenaltyController:
    """Q-learning агент для адаптивного управления штрафами"""
    def __init__(self, n_penalties=5):
        self.n_penalties = n_penalties
        self.penalty_multipliers = [1.0] * n_penalties
        self.loss_history = []

        # Q-learning parameters
        self.state_size = 5
        self.q_network = self._build_q_network()
        self.memory = []
        self.memory_size = 500
        self.batch_size = 8
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.q_network = self._build_q_network()
        self.train_step = 0

        # Сохраняем предыдущее состояние для обучения
        self.prev_state = None
        self.prev_multipliers = self.penalty_multipliers.copy()

        # ← ВСТАВЛЯЕМ ШАГ 3 ЗДЕСЬ
        self.recent_recompilation = 0  # 0 = нет недавней перекомпиляции

        # ← ДОБАВИТЬ ПОСЛЕ СУЩЕСТВУЮЩИХ ПОЛЕЙ
        self.test_probe_hit = False  # Флаг успешной тест-пробы
        self.test_probe_hit_bonus = 8.0  # Размер бонуса (может меняться)

        self.reward_lstm = self._build_reward_lstm()
        self.reward_memory = []
        self.reward_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.mixing_weight = 0.7
        self.mixing_lr = 0.005
        self.window_size = 20  # ← ДОБАВЬТЕ ЭТУ СТРОКУ

        

    def _build_q_network(self):
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.n_penalties, activation='linear')(x)
        return tf.keras.Model(inputs, outputs)

    def _get_state(self, current_loss, previous_losses):
        if len(previous_losses) < 3:
            return np.array([current_loss, 0, 0, 0, 0])

        recent = previous_losses[-3:]
        state = [
            current_loss,
            np.mean(recent) - np.mean(previous_losses[-6:-3]) if len(previous_losses) >= 6 else 0,
            np.std(recent) / (np.mean(recent) + 1e-9),
            min(recent),
            len(previous_losses) / 100.0
        ]
        return np.array(state[:self.state_size])

    def update_q_network(self, current_loss):
        if len(self.loss_history) < 3:
            self.loss_history.append(current_loss)
            return

        current_state = self._get_state(current_loss, self.loss_history)

        if self.prev_state is not None:
            reward = self._compute_reward(current_loss, self.loss_history[-1])

            if len(self.memory) >= self.memory_size:
                self.memory.pop(0)

            chosen_action = int(np.argmax(self.prev_multipliers))
            self.memory.append((
                self.prev_state,
                chosen_action,
                reward,
                current_state,
                False
            ))

            self._train_q_network()

        self.prev_state = current_state
        self.prev_multipliers = self.penalty_multipliers.copy()
        self.loss_history.append(current_loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _compute_reward(self, current_loss, previous_loss):
        if not hasattr(self, 'reward_lstm'):
            self.reward_lstm = self._build_reward_lstm()
            self.reward_memory = []
            self.window_size = 20
            self.reward_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.mixing_weight = 0.7
            self.mixing_lr = 0.005

        # ← ВСТАВЛЯЕМ ЗДЕСЬ ШАГ 1 КОД
        recompilation_bonus = 0.0
        if hasattr(self, 'recent_recompilation') and self.recent_recompilation > 0:
            if current_loss < previous_loss:
                improvement_ratio = (previous_loss - current_loss) / max(previous_loss, 1e-9)
                if improvement_ratio > 0.1:
                    recompilation_bonus = 30.0  # Увеличено
                    logging.info(f"🔥 СИЛЬНОЕ ПООЩРЕНИЕ после перекомпиляции: {recompilation_bonus}")
                elif improvement_ratio > 0.05:
                    recompilation_bonus = 20.0  # Среднее
                elif improvement_ratio > 0.01:
                    recompilation_bonus = 10.0  # Малое
            self.recent_recompilation = max(0, self.recent_recompilation - 1)
        

        features = np.array([
            current_loss,
            previous_loss,
            (previous_loss - current_loss) / max(previous_loss, 1e-9),
            len(self.loss_history) / 100.0,
            self.epsilon,
            np.mean(self.penalty_multipliers),
            1.0 if current_loss < previous_loss else 0.0
        ])

        self.reward_memory.append(features)
        if len(self.reward_memory) > self.window_size:
            self.reward_memory.pop(0)

        if len(self.reward_memory) >= self.window_size:
            sequence = np.array([self.reward_memory])

            if self.reward_lstm is None:
                self.reward_lstm = self._build_reward_lstm()
                self.reward_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            lstm_reward = self.reward_lstm.predict(sequence, verbose=0)[0][0] * 30.0
            improvement = previous_loss - current_loss
            if improvement > 0:
                base_reward = 10.0 * min(3.0, improvement / max(previous_loss, 1e-9))
            else:
                base_reward = -10.0 * min(3.0, abs(improvement) / max(previous_loss, 1e-9))

            if len(self.loss_history) > 20 and len(self.loss_history) % 5 == 0:
                recent_improvements = []
                for i in range(1, min(6, len(self.loss_history))):
                    past_loss = self.loss_history[-(i+1)]
                    current = self.loss_history[-i]
                    imp = (past_loss - current) / max(past_loss, 1e-9)
                    recent_improvements.append(imp)

                if recent_improvements:
                    lstm_performance = np.mean(recent_improvements)
                    if lstm_performance > 0.02:
                        self.mixing_weight = min(0.9, self.mixing_weight + self.mixing_lr)
                    elif lstm_performance < -0.01:
                        self.mixing_weight = max(0.3, self.mixing_weight - self.mixing_lr)

            reward = self.mixing_weight * lstm_reward + (1 - self.mixing_weight) * base_reward

            if len(self.loss_history) >= 3:
                self._update_reward_lstm(sequence, reward)

        else:
            improvement = previous_loss - current_loss
            improvement_ratio = improvement / max(previous_loss, 1e-9)

            if improvement_ratio > 0.1:
                reward = 30.0
            elif improvement_ratio > 0.05:
                reward = 15.0
            elif improvement_ratio > 0.01:
                reward = 5.0
            elif improvement > 0:
                reward = 1.0
            else:
                reward = -10.0 * min(3.0, abs(improvement) / max(previous_loss, 1e-9))

        # ← ДОБАВЛЯЕМ БОНУС ПЕРЕД ВОЗВРАТОМ
        if recompilation_bonus > 0:
            reward += recompilation_bonus

       # ← ДОБАВИТЬ ПРЯМО ПЕРЕД return (перед строкой с return)
        if hasattr(self, 'test_probe_hit') and self.test_probe_hit:
            bon = getattr(self, 'test_probe_hit_bonus', 8.0)
            reward += bon
            logging.info(f"➕ БОНУС за тест-пробу: +{bon}")
            self.test_probe_hit = False  # Сбрасываем флаг


        return float(np.clip(reward, -30.0, 30.0))

    def _build_reward_lstm(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 7)),
            tf.keras.layers.LSTM(12, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        return model

    def _update_reward_lstm(self, sequence, given_reward):
        if len(self.loss_history) < 3:
            return

        future_improvement = self.loss_history[-2] - self.loss_history[-1]
        target = 0.5 if future_improvement > 0 else -0.5

        with tf.GradientTape() as tape:
            prediction = self.reward_lstm(sequence, training=True)
            loss = tf.keras.losses.MSE(target, prediction)

        gradients = tape.gradient(loss, self.reward_lstm.trainable_variables)
        self.reward_optimizer.apply_gradients(zip(gradients, self.reward_lstm.trainable_variables))

    def _train_q_network(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            current_q_values = self.q_network(states)
            next_q_values = self.q_network(next_states)
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

            batch_indices = tf.range(tf.shape(current_q_values)[0])
            action_indices = tf.stack([batch_indices, actions], axis=1)
            chosen_q_values = tf.gather_nd(current_q_values, action_indices)

            loss = tf.keras.losses.MSE(target_q, chosen_q_values)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        if all(g is not None for g in gradients):
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        self.train_step += 1
        if self.train_step % 10 == 0:
            logging.debug(f"Q-agent training: loss={loss.numpy():.4f}, epsilon={self.epsilon:.3f}")

    def get_penalty_multipliers(self, current_loss, previous_losses):
        state = self._get_state(current_loss, previous_losses)
        state = np.expand_dims(state, axis=0)

        if np.random.random() < self.epsilon:
            multipliers = np.random.uniform(0.3, 2.0, self.n_penalties)
        else:
            if self.q_network is None:
                self.q_network = self._build_q_network()
            q_values = self.q_network.predict(state, verbose=0)[0]
            multipliers = 1.0 + 0.8 * np.tanh(q_values)

        self.penalty_multipliers = multipliers.tolist()
        logging.debug(f"Q-agent: множители {[f'{m:.2f}' for m in multipliers]}, epsilon: {self.epsilon:.3f}")
        return self.penalty_multipliers


class MultiHeadAttentionWithHeads(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, head_names, q_agent_embedding_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_names = head_names
        self.q_agent_embedding_dim = q_agent_embedding_dim
        
        self.attention_layers = {}

        self.embedding_dim = 128
        self.enriched_dim = self.embedding_dim + q_agent_embedding_dim
        self.q_agent_embedding = Dense(
            q_agent_embedding_dim,
            activation='tanh',
            name="q_agent_embedding"
        )

        self.head_embeddings = self.add_weight(
            shape=(len(self.head_names), self.enriched_dim),
            initializer="random_normal",
            trainable=True,
            name="head_embeddings"
        )
        
        self.head_index = {name: i for i, name in enumerate(self.head_names)}
        
        for head_name in head_names:
            self.attention_layers[head_name] = tf.keras.layers.MultiHeadAttention(
                num_heads=1, key_dim=key_dim, name=f"{head_name}_attention"
            )
        
        # Проекция и масштабирование для explanation головы
        if 'explanation' in head_names:
            
            
            self.explanation_proj = Dense(
                self.key_dim,
                activation='tanh',
                name="explanation_proj"
            )
            
            # Этот слой теперь проецирует из key_dim → enriched_dim
            self.explanation_scale = Dense(
                self.enriched_dim,
                activation='tanh',          # или None, если хотите чистый масштаб
                name="explanation_scale"
            )

    def build(self, input_shape):
        super().build(input_shape)
        self.q_agent_embedding.build((None, 5))
        
        enriched_shape = (input_shape[0], input_shape[1], input_shape[2] + self.q_agent_embedding_dim)
        
        for head_name in self.head_names:
            attention_layer = self.attention_layers[head_name]
            attention_layer.build(
                query_shape=enriched_shape,
                key_shape=enriched_shape,
                value_shape=enriched_shape
            )
        
        if hasattr(self, 'explanation_proj'):
            self.explanation_proj.build((None, self.enriched_dim))
        
        if hasattr(self, 'explanation_scale'):
            self.explanation_scale.build((None, self.key_dim))

        # ===== ДОБАВЛЯЕМ СОЗДАНИЕ ВЕСОВ ДЛЯ SYNTAX =====
        # Создаем обучаемые веса для синтаксической головы (макс. длина 150)
        self.syntax_pos_weights = self.add_weight(
            shape=(150, 1),
            initializer='ones',
            trainable=True,
            name='syntax_pos_weights'
        )

    def compute_mask(self, inputs, mask=None):
        return mask


    def call(self, inputs, mask=None, q_agent_state=None):
        seq_len = tf.shape(inputs)[1]
        attention_outputs = []

        if q_agent_state is not None:
            q_agent_embedded = self.q_agent_embedding(q_agent_state)
            q_agent_broadcast = tf.expand_dims(q_agent_embedded, axis=1)
            q_agent_broadcast = tf.tile(q_agent_broadcast, [1, seq_len, 1])
            enriched_inputs = tf.concat([inputs, q_agent_broadcast], axis=-1)
        else:
            enriched_inputs = inputs

        for head_name in self.head_names:
            head_index = self.head_index[head_name]
            head_bias = self.head_embeddings[head_index]
            head_bias = head_bias[None, None, :]
            expert_input = enriched_inputs + head_bias
            
            attn_output = self.attention_layers[head_name](
                expert_input, expert_input, attention_mask=mask
            )
        
            # Минимальный explain-модулятор — исправленная версия
            if head_name == 'explanation' and hasattr(self, 'explanation_proj'):
                # ИСПРАВЛЕНО: используем существующую маску
                if mask is not None:
                   
                    seq_mask = mask[:, :, :, 0]      
                    seq_mask = tf.transpose(seq_mask, [0, 2, 1])  
                else:
                    
                    seq_mask = tf.ones_like(enriched_inputs[:, :, :1])
            
                # Вычисляем среднее только по реальным токенам
                summed = tf.reduce_sum(enriched_inputs * seq_mask, axis=1, keepdims=True)
                count = tf.reduce_sum(seq_mask, axis=1, keepdims=True)
                context = summed / (count + 1e-9)  # (b, 1, enriched_dim)
            
                context_flat = tf.squeeze(context, axis=1)  # (b, enriched_dim)
            
                expl_bias = self.explanation_proj(context_flat)  # (b, key_dim)
            
                expl_scale = tf.tanh(self.explanation_scale(expl_bias))  # (b, enriched_dim)
                expl_scale = expl_scale[:, None, :]  # (b, 1, enriched_dim)
            
                # Применяем мягкое масштабирование
                attn_output = attn_output * (1.0 + 0.75 * expl_scale)

            # ===== НОВОЕ: УСИЛЕНИЕ ГОЛОВЫ SYNTAX =====
            if head_name == 'syntax':
                
                seq_len = tf.shape(attn_output)[1]
                # Берём веса для текущей длины
                pos_weights = self.syntax_pos_weights[:seq_len]  # (seq_len, 1)
                pos_weights = pos_weights[tf.newaxis, :, :]      # (1, seq_len, 1)
                
                attn_output = attn_output * (1.0 + 0.3 * pos_weights)

            attention_outputs.append(attn_output)

        combined = tf.add_n(attention_outputs)
        combined = combined / float(len(attention_outputs))
        return combined


    def compute_output_shape(self, input_shape):
        enriched_dim = input_shape[-1] + self.q_agent_embedding_dim
        return (input_shape[0], input_shape[1], enriched_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'head_names': self.head_names,
            'q_agent_embedding_dim': self.q_agent_embedding_dim,
            'has_explanation_proj': 'explanation' in self.head_names,
        })
        return config

    @classmethod
    def from_config(cls, config):
        has_expl_proj = config.pop('has_explanation_proj', False)
        instance = cls(**config)
        
        if has_expl_proj and not hasattr(instance, 'explanation_proj'):
            instance.explanation_proj = Dense(
                instance.key_dim,
                activation='tanh',
                name="explanation_proj"
            )
            instance.explanation_scale = Dense(
                128 + instance.q_agent_embedding_dim,   # явно указываем enriched_dim
                activation='tanh',
                name="explanation_scale"
            )
        
        return instance

    
class AttentionController(tf.keras.layers.Layer):
    """Стандартный контролёр масштабирования после attention"""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_dim,
            activation='tanh',
            name="attention_controller"
        )

        # Важно: сообщаем Keras, что слой поддерживает маски
        self.supports_masking = True

    def call(self, x, q_agent_state):
        scale = self.dense(q_agent_state)           # (B, D)
        scale = tf.expand_dims(scale, axis=1)       # (B, 1, D)
        return x * (1.0 + scale)


class QPenalizedCrossentropy(tf.keras.losses.Loss):
    def __init__(self, penalty_controller, base_loss, **kwargs):
        super().__init__(**kwargs)
        self.penalty_controller = penalty_controller
        self.base_loss = base_loss
        self.previous_ce = tf.Variable(10.0, trainable=False, dtype=tf.float32)
        self.steps = tf.Variable(0, trainable=False, dtype=tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'penalty_controller': None,  # не сериализуем
            'base_loss': self.base_loss,  # сериализуем
        })
        return config

    @classmethod
    def from_config(cls, config):
        # base_loss может быть строкой или объектом — обработаем
        base_loss = config.pop('base_loss', None)
        config.pop('penalty_controller', None)  # не нужен при загрузке
        
        if base_loss is None:
            base_loss = tf.keras.losses.CategoricalCrossentropy()
        elif isinstance(base_loss, dict):
            # Пытаемся десериализовать
            base_loss = tf.keras.losses.deserialize(base_loss)
        
        # penalty_controller будет заменён при компиляции — временно None
        instance = cls(penalty_controller=None, base_loss=base_loss, **config)
        return instance
    

    def call(self, y_true, y_pred):
        ce_loss = self.base_loss(y_true, y_pred)
        current_ce = tf.reduce_mean(ce_loss)

        self.steps.assign_add(1)

        skip_initial = self.steps < 10
        improvement = self.previous_ce - current_ce
        self.previous_ce.assign(current_ce)

        if hasattr(self.penalty_controller, 'penalty_multipliers'):
            multipliers = self.penalty_controller.penalty_multipliers
            avg_multiplier = sum(multipliers) / len(multipliers)
            excess = avg_multiplier - 1.0

            should_penalize = tf.logical_and(
                tf.logical_not(skip_initial),
                tf.logical_and(
                    improvement <= 0.0,
                    excess > 0.0
                )
            )

            penalty_boost = 0.5 * excess
            penalty_boost = tf.clip_by_value(penalty_boost, 0.0, 0.25)

            penalised_loss = ce_loss * (1.0 + penalty_boost)
            return tf.where(should_penalize, penalised_loss, ce_loss)

        return ce_loss


class QTextNetwork:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.model = None
        # Инициализация морфологического анализатора
        try:
            import pymorphy2
            self.morph = pymorphy2.MorphAnalyzer()
            logging.info("pymorphy2 успешно загружен")
        except ImportError:
            self.morph = None
            logging.warning("pymorphy2 не установлен. Лемматизация отключена.")
        self.tokenizer = None
        self.max_sequence_length = 150
        self.vocab_size = 50000
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

        # === ID служебных слов ===
        self.stopword_ids = set()

        self.attention_head_names = ['semantic', 'syntax', 'ngram_pattern', 'causal', 'coordinator', 'corpus_order', 'explanation']

        self.base_causal_words = {
            'потому что', 'поэтому', 'так как', 'поскольку', 'если', 'из-за',
            'благодаря', 'вследствие', 'в результате', 'причина', 'следствие',
            'результат', 'основание', 'фактор', 'условие'
        }
        self.learned_causal_words = set()
        self.causal_confidence = {}
        self.causal_patterns = []

        self.already_switched_to_topk = False
        self.current_window_ratio = 0.05
        self.last_hit_rate = 0.7

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/app.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.qa_pairs_list = []
        self.previous_losses = []

        self.penalty_controller = QPenaltyController(n_penalties=4)
        self._model_built = False
        self.current_training_loss = 1.0
        self.model_compiled = False
        self.is_trained = False

    def create_strong_target(self, correct_word_indices, confidence=0.98):
        if not correct_word_indices:
            correct_word_indices = [1]

        target = np.zeros(self.vocab_size, dtype=np.float32)
        prob_per_word = confidence / len(correct_word_indices)

        for idx in correct_word_indices:
            if 0 < idx < self.vocab_size:
                target[idx] = prob_per_word

        target += (1.0 - confidence) / self.vocab_size
        return target

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

            
            unique_words = len(set(' '.join(self.corpus_data).split()))
            
            logging.info(f"Обновлен размер словаря: {self.vocab_size}")


            self.query_answer_pairs = self.load_all_query_answer_pairs()
            self.qa_pairs_list = self.query_answer_pairs
            logging.info(f"Загружено пар вопрос-ответ: {len(self.query_answer_pairs)}")

            if len(self.query_answer_pairs) == 0:
                logging.warning("Нет пар вопрос-ответ для обучения")
                return False

            if not hasattr(self, 'previous_losses') or self.previous_losses is None:
                self.previous_losses = []
            if self.penalty_controller is None:
                self.penalty_controller = QPenaltyController(n_penalties=4)

            # Добавить проверку сетей в penalty_controller
            if self.penalty_controller.q_network is None:
                self.penalty_controller.q_network = self.penalty_controller._build_q_network()
            if self.penalty_controller.reward_lstm is None:
                self.penalty_controller.reward_lstm = self.penalty_controller._build_reward_lstm()
                self.penalty_controller.reward_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            self._build_corpus_chains()
            self._extract_keyword_chains()
            self._build_chain_vectors()
            logging.info(f"Сформировано цепочек ключевых слов: {len(self.keyword_chains)}")

            self.extract_causal_patterns()

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

                    # 1. БИГРАММЫ - только если соседние слова разные
                    for i in range(len(words) - 1):
                        if words[i] != words[i+1]:  # ← ДОБАВЛЕНО: проверка на разные слова
                            bigram = ' '.join(words[i:i+2])
                            self.corpus_bigrams[bigram] += 1

                    # 2. ТРИГРАММЫ - только если все три слова разные
                    for i in range(len(words) - 2):
                        word1, word2, word3 = words[i], words[i+1], words[i+2]
                        if word1 != word2 and word2 != word3 and word1 != word3:  # ← ДОБАВЛЕНО: проверка
                            trigram = ' '.join(words[i:i+3])
                            self.corpus_trigrams[trigram] += 1

        logging.info(f"Сформировано последовательностей корпуса: {len(self.corpus_sequences)}")
        logging.info(f"Уникальных слов: {len(self.corpus_stats)}")
        logging.info(f"Уникальных биграмм: {len(self.corpus_bigrams)}")
        logging.info(f"Уникальных триграмм: {len(self.corpus_trigrams)}")

        if self.corpus_stats:
            
            total_unique = len(self.corpus_stats)
            frequent_words = sum(1 for count in self.corpus_stats.values() if count > 2)  # БЫЛО >1

            
            if frequent_words < 5000:
                optimal = min(total_unique, 15000)
            else:
                optimal = int(frequent_words * 1.15)

            max_growth = int(self.vocab_size * 1.2)
            optimal = max(8000, min(optimal, max_growth, 45000))  # Добавили max_growth
            self.vocab_size = max(optimal, 10000)

            coverage = frequent_words / total_unique if total_unique > 0 else 0
            logging.info(f"Оптимальный размер словаря: {optimal}")
            logging.info(f"Частых слов (>1 раз): {frequent_words} ({coverage:.1%} от {total_unique} уникальных)")

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
        matching_sentences = []
        words_lower = [word.lower() for word in words]

        for text in self.corpus_data:
            if isinstance(text, str) and text.strip():
                text_lower = text.lower()
                if any(word in text_lower for word in words_lower):
                    matching_sentences.append(text)

        logging.info(f"Найдено предложений с словами {words}: {len(matching_sentences)}")
        return matching_sentences

    def extract_semantic_core(self, text, min_common_words=1):
        words = text.lower().split()
        word_counts = Counter(words)
        semantic_core = [word for word, count in word_counts.items() if count >= min_common_words]
        return semantic_core

    def word_in_pseudo_answers(self, word):
        for pair in self.query_answer_pairs:
            if word.lower() in pair['answer'].lower().split():
                return True
        return False

    def word_pair_in_pseudo_answers(self, word_pair):
        pair_text = ' '.join(word_pair).lower()
        for pair in self.query_answer_pairs:
            if pair_text in pair['answer'].lower():
                return True
        return False

    def filter_relevant_words(self, words, context):
        context_words = context.lower().split()
        return [word for word in words if self.is_semantically_related(word, context_words)]

    def is_semantically_related(self, word, context_words):
        word = word.lower()
        if word in context_words or self.word_in_pseudo_answers(word):
            return True
        for context_word in context_words:
            if self.word_pair_in_pseudo_answers([word, context_word]):
                return True
        return False

    def extract_causal_patterns(self):
        logging.info("Извлечение каузальных паттернов из корпуса...")

        causal_indicators = [
            'потому что', 'поэтому', 'так как', 'в результате',
            'следовательно', 'из-за', 'благодаря', 'вследствие',
            'приводит к', 'вызывает', 'влечет', 'обусловливает'
        ]

        for text in self.corpus_data:
            if not isinstance(text, str):
                continue

            text_lower = text.lower()

            for connector in causal_indicators:
                if connector in text_lower:
                    parts = text.split(connector)
                    if len(parts) >= 2:
                        cause_part = parts[0].strip()
                        effect_part = parts[1].strip()

                        cause_words = self.extract_keywords(cause_part)
                        effect_words = self.extract_keywords(effect_part)

                        if cause_words and effect_words:
                            pattern = {
                                'cause': cause_words[:3],
                                'effect': effect_words[:3],
                                'connector': connector
                            }
                            self.causal_patterns.append(pattern)

                            self.learned_causal_words.update(cause_words[:3])
                            self.learned_causal_words.update(effect_words[:3])

        logging.info(f"Извлечено каузальных паттернов: {len(self.causal_patterns)}")
        logging.info(f"Найдено каузальных слов: {len(self.learned_causal_words)}")

    def is_word_statistically_significant(self, current_words, candidate_word):
        """
        Проверяет, является ли кандидат статистически значимым продолжением
        в контексте предыдущих слов на основе частот N-грамм.
        """

        if not candidate_word:
            return False

        candidate = candidate_word.lower()

        # === 1. БЫСТРЫЕ ПРОПУСКИ ===

        # Частые слова
        if self.corpus_stats.get(candidate, 0) >= 50:
            return True

        # Стоп-слова
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'о', 'от', 'до', 'не', 'но',
            'из', 'у', 'за', 'а', 'то', 'это', 'что', 'как',
            'для', 'или', 'бы', 'ли', 'вот', 'так'
        }
        if candidate in stop_words or len(candidate) <= 2:
            return True

        if not current_words:
            return True

        context = [w.lower() for w in current_words]

        # === ПАРАМЕТРЫ ===
        MIN_BIGRAM_FREQ = 2
        MIN_TRIGRAM_FREQ = 1
        MIN_WORD_FREQ = 3

        # === 2. ТРИГРАММЫ (СИЛЬНЕЙШИЙ СИГНАЛ) ===
        if len(context) >= 2:
            trigram = f"{context[-2]} {context[-1]} {candidate}"
            prev_bigram = f"{context[-2]} {context[-1]}"

            trigram_freq = self.corpus_trigrams.get(trigram, 0)
            prev_bigram_freq = self.corpus_bigrams.get(prev_bigram, 0)

            if (
                trigram_freq >= MIN_TRIGRAM_FREQ and
                prev_bigram_freq >= 1 and
                trigram_freq / prev_bigram_freq >= 0.001
            ):
                return True

        # === 3. БИГРАММЫ ===
        last_word = context[-1]
        bigram = f"{last_word} {candidate}"

        bigram_freq = self.corpus_bigrams.get(bigram, 0)
        last_word_freq = self.corpus_stats.get(last_word, 0)

        if (
            bigram_freq >= MIN_BIGRAM_FREQ and
            last_word_freq >= MIN_WORD_FREQ and
            bigram_freq / last_word_freq >= 0.001
        ):
            return True

        return False



    def extract_keywords(self, text, max_words=3):
        words = text.lower().split()
        if len(words) < 2:
            return words

        stop_words = {'и', 'в', 'на', 'с', 'по', 'о', 'от', 'до', 'не', 'но', 'из', 'у', 'за', 'же'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        return filtered_words[:max_words]

    def update_causal_vocabulary(self, generated_text, attention_weights, reward):
        if reward < 0.7:
            return

        words = generated_text.lower().split()

        for word in words:
            if len(word) < 3 or word in self.base_causal_words:
                continue

            word_idx = words.index(word) if word in words else -1
            if word_idx > 0:
                left_context = words[max(0, word_idx-2):word_idx]
                right_context = words[word_idx+1:min(len(words), word_idx+3)]

                left_has_causal = any(causal_word in left_context for causal_word in self.base_causal_words)
                right_has_causal = any(causal_word in right_context for causal_word in self.base_causal_words)

                if left_has_causal or right_has_causal:
                    self.learned_causal_words.add(word)
                    confidence = min(reward * 1.2, 0.9)
                    self.causal_confidence[word] = confidence
                    logging.info(f"Добавлено каузальное слово: '{word}' (уверенность: {confidence:.2f})")

    def apply_causal_boost(self, base_probs, current_context):
        boosted_probs = base_probs.copy()
        current_text_lower = current_context.lower()

        all_causal_words = self.base_causal_words.union(self.learned_causal_words)

        boost_factor = 1.3

        for word, word_idx in self.tokenizer.word_index.items():
            if word_idx >= len(boosted_probs):
                continue

            if word in all_causal_words:
                if self.is_context_relevant_for_causal(word, current_text_lower):
                    boosted_probs[word_idx] *= boost_factor

        if np.sum(boosted_probs) > 0:
            boosted_probs = boosted_probs / np.sum(boosted_probs)

        return boosted_probs

    def is_context_relevant_for_causal(self, word, context):
        question_words = {'почему', 'зачем', 'как', 'что', 'когда'}
        if any(q_word in context for q_word in question_words):
            return True
        if any(marker in context for marker in self.base_causal_words):
            return True
        return False

    def evaluate_causal_quality(self, text):
        text_lower = text.lower()
        words = set(text_lower.split())

        connectors_score = 0
        causal_connectors = ['потому что', 'поэтому', 'так как', 'следовательно']
        for connector in causal_connectors:
            if connector in text_lower:
                connectors_score += 0.25

        all_causal_words = self.base_causal_words.union(self.learned_causal_words)
        causal_word_count = len(words.intersection(all_causal_words))
        words_score = min(causal_word_count / 3.0, 0.5)

        patterns_score = 0
        for pattern in self.causal_patterns[:5]:
            has_cause = any(cause_word in text_lower for cause_word in pattern['cause'])
            has_effect = any(effect_word in text_lower for effect_word in pattern['effect'])
            if has_cause and has_effect:
                patterns_score += 0.1

        total_score = connectors_score + words_score + patterns_score
        return min(total_score, 1.0)

    def find_best_matching_answer(self, generated_text):
        best_similarity = 0
        best_answer = ""

        for pair in self.query_answer_pairs:
            answer_text = pair['answer'].lower()
            generated_words = set(generated_text.lower().split())
            answer_words = set(answer_text.split())

            overlap = len(generated_words.intersection(answer_words))
            total_unique = len(generated_words.union(answer_words))

            if total_unique > 0:
                similarity = overlap / total_unique
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_answer = pair['answer']

        return best_answer, best_similarity

    def clean_generated_text(self, generated_text, reference_answer):
        if not reference_answer:
            return generated_text

        ref_words = set(reference_answer.lower().split())
        generated_words = generated_text.split()

        cleaned_words = [word for word in generated_words if word.lower() in ref_words]

        if not cleaned_words:
            return generated_text

        return ' '.join(cleaned_words)

    #def generate_multiple_variants(self, seed_words, num_variants=15, max_length=30):
    def generate_multiple_variants(self, seed_words, num_variants=15, max_length=30, return_only_new_tokens=False):
        
        variants = []

        for i in range(num_variants):
            temperature = 0.7 + (i * 0.6/num_variants)

            if isinstance(seed_words, list):
                seed_text = ' '.join(seed_words)
            else:
                seed_text = seed_words

            variant, confidence = self.generate_text_from_seed(
                seed_text,
                max_length=max_length,
                temperature=temperature,
                return_only_new_tokens=return_only_new_tokens
            )
            variants.append((variant, confidence))

        return variants

    def select_best_variant(self, variants):
        best_variant = ""
        best_similarity = 0
        best_confidence = 0

        for variant, confidence in variants:
            best_answer, similarity = self.find_best_matching_answer(variant)
            combined_score = (similarity + confidence) / 2

            if combined_score > best_similarity:
                best_similarity = combined_score
                best_variant = variant
                best_confidence = confidence

        best_answer, _ = self.find_best_matching_answer(best_variant)
        cleaned_variant = self.clean_generated_text(best_variant, best_answer)

        return cleaned_variant, best_confidence

    def get_enhanced_chain_reward(self, pseudo_question, pseudo_answer, generated_chain):
        if not pseudo_question or not pseudo_answer or not generated_chain:
            return 0.0
        if not hasattr(self, 'previous_losses') or self.previous_losses is None:
            self.previous_losses = []

        question_words = pseudo_question.split()
        answer_words = pseudo_answer.split()
        generated_words = generated_chain.split()

        question_word_match = len(set(question_words).intersection(generated_words)) / max(len(question_words), 1)
        answer_word_match = len(set(answer_words).intersection(generated_words)) / max(len(answer_words), 1)
        question_bigrams = [' '.join(question_words[i:i+2]) for i in range(len(question_words)-1)]
        answer_bigrams = [' '.join(answer_words[i:i+2]) for i in range(len(answer_words)-1)]
        generated_bigrams = [' '.join(generated_words[i:i+2]) for i in range(len(generated_words)-1)]
        bigram_question_match = len(set(question_bigrams).intersection(generated_bigrams)) / max(len(question_bigrams), 1)
        bigram_answer_match = len(set(answer_bigrams).intersection(generated_bigrams)) / max(len(answer_bigrams), 1)
        question_trigrams = [' '.join(question_words[i:i+3]) for i in range(len(question_words)-2)]
        answer_trigrams = [' '.join(answer_words[i:i+3]) for i in range(len(answer_words)-2)]
        generated_trigrams = [' '.join(generated_words[i:i+3]) for i in range(len(generated_words)-2)]
        trigram_question_match = len(set(question_trigrams).intersection(generated_trigrams)) / max(len(question_trigrams), 1)
        trigram_answer_match = len(set(answer_trigrams).intersection(generated_trigrams)) / max(len(answer_trigrams), 1)
        word_counts = Counter(generated_words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_penalty = repeated_words / len(generated_words) if generated_words else 0
        causal_score = self.evaluate_causal_quality(generated_chain)

        base_semantic_penalty = 0.5
        base_syntax_penalty = 0.4
        base_ngram_penalty = 0.3
        base_causal_penalty = 0.6

        apply_semantic_penalty = not (question_word_match > 0.1 or answer_word_match > 0.1)
        apply_syntax_penalty = not (bigram_question_match > 0.05 or bigram_answer_match > 0.05)
        apply_ngram_penalty = not (trigram_question_match > 0.02 or trigram_answer_match > 0.02)
        apply_causal_penalty = not (causal_score > 0.1)

        if hasattr(self, 'penalty_controller') and self.penalty_controller:
            current_loss = getattr(self, 'current_training_loss', 1.0)
            multipliers = self.penalty_controller.get_penalty_multipliers(
                current_loss,
                self.previous_losses
            )
        else:
            multipliers = [1.0, 1.0, 1.0, 1.0]

        semantic_penalty = base_semantic_penalty * multipliers[0] if apply_semantic_penalty else 0.0
        syntax_penalty = base_syntax_penalty * multipliers[1] if apply_syntax_penalty else 0.0
        ngram_penalty = base_ngram_penalty * multipliers[2] if apply_ngram_penalty else 0.0
        causal_penalty = base_causal_penalty * multipliers[3] if apply_causal_penalty else 0.0

        existing_reward = (
            0.35 * question_word_match +
            0.20 * answer_word_match +
            0.10 * bigram_question_match +
            0.10 * bigram_answer_match +
            0.05 * trigram_question_match +
            0.05 * trigram_answer_match +
            0.15 * causal_score
        )

        existing_reward *= (1 - repetition_penalty * 0.3)
        existing_reward = existing_reward * (1 - semantic_penalty) * (1 - syntax_penalty) * (1 - ngram_penalty) * (1 - causal_penalty)

        question_topic_words = set(pseudo_question.split())
        generated_topic_words = set(generated_chain.split())
        stop_words = {'ты', 'какой', 'что', 'как', 'это', 'тот', 'такой'}
        question_content = question_topic_words - stop_words
        generated_content = generated_topic_words - stop_words
        if question_content:
            topic_overlap = len(question_content.intersection(generated_content))
            topic_penalty = 1.0 - (topic_overlap / max(len(question_content), 1))
        else:
            topic_penalty = 0.5

        # ===== НОВОЕ: МЕТРИКА ПОРЯДКА СЛОВ (БЕЗ МОРФОЛОГИИ) =====
        gen_words_lower = generated_chain.lower().split()
        if len(gen_words_lower) >= 2:
            bigram_matches = 0
            total_bigrams = len(gen_words_lower) - 1
            for i in range(total_bigrams):
                bigram = ' '.join(gen_words_lower[i:i+2])
                if self.corpus_bigrams.get(bigram, 0) > 0:
                    bigram_matches += 1
            order_score = bigram_matches / total_bigrams
            # Дополнительный бонус за частотность биграмм
            freqs = [self.corpus_bigrams.get(' '.join(gen_words_lower[i:i+2]), 0) 
                     for i in range(total_bigrams) if self.corpus_bigrams.get(' '.join(gen_words_lower[i:i+2]), 0) > 0]
            if freqs:
                avg_freq = np.mean(freqs) / 10.0
                order_score = 0.7 * order_score + 0.3 * min(1.0, avg_freq)
        else:
            order_score = 0.0

        total_reward = (existing_reward * 0.45) + (topic_penalty * 0.25) + (order_score * 0.30)
        return min(total_reward, 1.0)

    def build_model(self):
        logging.info("Строительство улучшенной модели Q-сети с многоголовым трансформером...")

        inputs = Input(shape=(self.max_sequence_length,))
        q_agent_state_input = Input(shape=(5,), name="q_agent_state")
        x = Embedding(input_dim=self.vocab_size, output_dim=128)(inputs)

        positions = tf.range(start=0, limit=self.max_sequence_length, delta=1)
        positions = tf.expand_dims(positions, 0)
        pos_encoding = tf.keras.layers.Embedding(
            input_dim=self.max_sequence_length,
            output_dim=128
        )(positions)
        x = x + pos_encoding

        # ===== МИНИМАЛЬНОЕ ИЗМЕНЕНИЕ: добавить квадратичные признаки и спроецировать обратно =====
        x_squared = tf.keras.layers.Lambda(
            _square_features,
            output_shape=lambda input_shape: (input_shape[0], input_shape[1], input_shape[2] * 2)
        )(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x_squared)  # проекция обратно на 128
        # ===== КОНЕЦ ИЗМЕНЕНИЯ =====

        
        mask = tf.keras.layers.Lambda(
            _pad_mask,
            output_shape=lambda input_shape: input_shape
        )(inputs)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
    
        # Создаём динамическую causal mask
        causal_mask = tf.keras.layers.Lambda(
            _causal_mask,
            output_shape=lambda input_shape: (input_shape[0], 1, 1, input_shape[1], input_shape[1]),
            name="dynamic_causal_mask"
        )(inputs)

        # Комбинируем маски
        mask = tf.keras.layers.Lambda(
            _combine_masks,
            output_shape=lambda input_shapes: input_shapes[0]
        )([mask, causal_mask])

        # ===== ПЕРВЫЙ MULTI-HEAD ATTENTION =====
        multi_head_attn = MultiHeadAttentionWithHeads(
            num_heads=len(self.attention_head_names),
            key_dim=32,
            head_names=self.attention_head_names,
            q_agent_embedding_dim=32,
            name="multi_head_attention_with_q_agent_v2"
        )

        attention_output = multi_head_attn(x, mask=mask, q_agent_state=q_agent_state_input)

        # Проекция размерности
        if attention_output.shape[-1] != 128:
            combined_attention_projected = Dense(128, name="attention_projection")(attention_output)
        else:
            combined_attention_projected = attention_output

        # AttentionController
        attention_controller = AttentionController(hidden_dim=128, name="attention_controller")
        combined_attention_projected = attention_controller(combined_attention_projected, q_agent_state_input)

        # ===== УЛУЧШЕННЫЙ TRANSFORMER BLOCK (pre-norm) =====
        # Нормализуем перед attention
        x = LayerNormalization(epsilon=1e-6, name="attn_pre_norm")(x)
        # Добавляем attention output
        x = x + combined_attention_projected
        # Вторая нормализация
        x = LayerNormalization(epsilon=1e-6, name="attn_post_norm")(x)

        # ===== ПЕРВЫЙ FFN БЛОК (параллельные головы) =====
        ff_outputs = []
        for i in range(len(self.attention_head_names)):
            ff = Dense(256, activation='relu', name=f"ff_{i}_dense1")(x)
            ff = Dropout(0.1, name=f"ff_{i}_drop")(ff)
            ff = Dense(128, name=f"ff_{i}_dense2")(ff)
            ff_outputs.append(ff)

        combined_ff = Concatenate(name="combined_ff")(ff_outputs)
        combined_ff_projected = Dense(128, name="ff_projection")(combined_ff)
        x = LayerNormalization(epsilon=1e-6, name="ff_norm")(x + combined_ff_projected)

        # ===== ВТОРОЙ ATTENTION БЛОК (остается без изменений) =====
        attention_outputs2 = []
        for head_name in self.attention_head_names:
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=32, name=f"{head_name}_attention_2"
            )
            attn_output = attn_layer(x, x, attention_mask=mask)
            attention_outputs2.append(attn_output)

        # Сначала объединяем головы (без весов)
        combined_attention2_temp = tf.keras.layers.Add()(attention_outputs2)
        combined_attention2_temp = Dense(128, name="attention_projection2_temp")(combined_attention2_temp)

        # Обновляем x с временным объединением
        x_temp = LayerNormalization()(x + combined_attention2_temp)

        # ТЕПЕРЬ вычисляем веса на основе обновленного x
        last_token = x_temp[:, -1, :]  # ← теперь x после attention
        coordinator_weights2 = tf.keras.layers.Dense(
            len(self.attention_head_names), activation='softmax', name='coordinator_weights2'
        )(last_token)

        # Адаптация весов
        if hasattr(self, 'already_switched_to_topk'):
            hit_rate = getattr(self, 'last_hit_rate', 0.7)
            if hit_rate < 0.6:
                boost_mask = tf.constant([[1.5, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0]])
                coordinator_weights2 = coordinator_weights2 * boost_mask
                coordinator_weights2 = tf.nn.softmax(coordinator_weights2)

        # Взвешиваем головы
        weighted_heads2 = []
        for i, head_output in enumerate(attention_outputs2):
            weight = coordinator_weights2[:, i:i+1, tf.newaxis]
            weighted_heads2.append(head_output * weight)

        # Финальное объединение
        combined_attention2 = tf.keras.layers.Add()(weighted_heads2)
        combined_attention2 = Dense(128, name="attention_projection2")(combined_attention2)
        x = LayerNormalization()(x_temp + combined_attention2)  # финальный x

        # ===== ВТОРОЙ FFN БЛОК =====
        ff2 = Dense(512, activation='relu')(x)
        ff2 = Dropout(0.15)(ff2)
        ff2 = Dense(128)(ff2)
        x = LayerNormalization()(x + ff2)

        # ===== ФИНАЛЬНЫЙ КЛАССИФИКАТОР =====
        last_token_output = x[:, -1, :]
        
        pooled = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=False)
        )(x)  # (batch, 256) 
        x_combined = Concatenate()([last_token_output, pooled])

        x = Dense(768, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                  bias_regularizer=tf.keras.regularizers.l2(0.01))(x_combined)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

        outputs = Dense(self.vocab_size, activation='softmax')(x)

        self.model = Model([inputs, q_agent_state_input], outputs)
        logging.info("Улучшенная модель построена успешно.")
        self.compile_model()
        self._model_built = True

    # СРАЗУ ПОСЛЕ ЭТОГО добавить метод transfer_weights():
    def transfer_weights(self, old_model, new_vocab_size):
        """Переносит веса со старой модели на новую с другим vocab_size"""
        old_vocab_size = old_model.output_shape[-1]
        
        if old_vocab_size == new_vocab_size:
            return True
        
        self.model_compiled = False   # ← Добавить эту строку 
        self.build_model()
        
        new_layers = self.model.layers[:-1]
        old_layers = old_model.layers[:-1]
        
        for new_layer, old_layer in zip(new_layers, old_layers):
            if len(new_layer.get_weights()) == len(old_layer.get_weights()):
                try:
                    new_layer.set_weights(old_layer.get_weights())
                except:
                    pass
        
        new_weights = self.model.layers[-1].get_weights()
        old_weights = old_model.layers[-1].get_weights()
        
        min_vocab = min(old_vocab_size, new_vocab_size)
        new_weights[0][:, :min_vocab] = old_weights[0][:, :min_vocab]
        new_weights[1][:min_vocab] = old_weights[1][:min_vocab]
        
        self.model.layers[-1].set_weights(new_weights)
        logging.info(f"Веса перенесены: {old_vocab_size} -> {new_vocab_size} токенов")
        return True

    
    #def generate_text_from_seed(self, seed_sequence, max_length=30, temperature=1.0):
    def generate_text_from_seed(self, seed_sequence, max_length=30, temperature=1.0, return_only_new_tokens=False):
        try:
            if isinstance(seed_sequence, list):
                seed_text = ' '.join(seed_sequence)
            else:
                seed_text = seed_sequence

            if not seed_text or len(seed_text.strip()) == 0:
                seed_text = random.choice([w for w, c in self.corpus_stats.items() if c > 50 and len(w) > 2])

            # Запоминаем исходное количество слов
            original_words = seed_text.lower().split()
            words = original_words.copy()

            for _ in range(max_length):
                # Оставляем место для генерации (например, 20 токенов)
                safe_limit = self.max_sequence_length - 20 
                if len(words) > safe_limit:
                    words = words[-safe_limit:]
                
                encoded = self.tokenizer.texts_to_sequences([' '.join(words)])
                encoded = pad_sequences(encoded, maxlen=self.max_sequence_length)

                q_agent_state = np.zeros((1, 5))

                if self.model is None:
                    self.build_model()
                    self._model_built = True
                preds = self.model.predict([encoded, q_agent_state], verbose=0)[0]

                preds = np.asarray(preds).astype('float64')
                preds = np.maximum(preds, 1e-12)                # <-- ДОБАВИТЬ НОВУЮ СТРОКУ
                preds = preds ** (1.0 / temperature)            # <-- НОВЫЙ temperature
                preds = preds / (np.sum(preds) + 1e-12)         # <-- НОВАЯ нормализация


                # ПРАВКА №1
                if len(words) > 0:
                    last_word = words[-1]
                    last_idx = self.tokenizer.word_index.get(last_word, None)
                    if last_idx is not None and last_idx < len(preds):
                        preds[last_idx] *= 0.4

                # ПРАВКА №2
                stopwords = {'что', 'быть', 'и', 'ли', 'как', 'это', 'то', 'в', 'на', 'с', 'по', 'о', 
                             'не', 'но', 'из', 'у', 'за', 'же', 'а', 'для', 'или', 'бы', 'вот', 'так'}
                for word in stopwords:
                    idx = self.tokenizer.word_index.get(word, None)
                    if idx is not None and idx < len(preds):
                        preds[idx] *= 0.3 #давим стоп слова при генерации

                # ПРАВКА №4
                if len(words) >= 3:
                    recent_words = words[-5:] if len(words) >= 5 else words
                    word_counts = {}
                    for w in recent_words:
                        word_counts[w] = word_counts.get(w, 0) + 1
    
                    for w, count in word_counts.items():
                        if count > 1:
                            idx = self.tokenizer.word_index.get(w, None)
                            if idx is not None and idx < len(preds):
                                preds[idx] *= (0.7 ** (count - 1))

                #preds = np.asarray(preds).astype('float64')

                if hasattr(self, 'already_switched_to_topk'):
                    current_context = ' '.join(words)
                    semantic_core = self.extract_semantic_core(current_context, min_common_words=1)
                    for word in semantic_core[:2]:
                        if word in self.tokenizer.word_index:
                            idx = self.tokenizer.word_index[word]
                            if 0 < idx < len(preds):
                                preds[idx] *= 1.2
                

                preds = self.apply_causal_boost(preds, ' '.join(words))

                # === АНТИ-ПОВТОР ПОСЛЕДНЕГО СЛОВА ===
                if len(words) > 0:
                    last_word = words[-1]
                    last_idx = self.tokenizer.word_index.get(last_word, None)
                    if last_idx is not None and last_idx < len(preds):
                        preds[last_idx] *= 0.4


                # === НОРМАЛИЗАЦИЯ ПОСЛЕ ВСЕХ ФИЛЬТРОВ ===
                if np.sum(preds) > 0:
                    preds = preds / np.sum(preds)
                # === КОНЕЦ НОРМАЛИЗАЦИИ ===


                # === ОГРАНИЧЕНИЕ ГЕНЕРАЦИИ TOP-K ОКНОМ ===
                if hasattr(self, 'already_switched_to_topk') and hasattr(self, 'topk_loss'):
                    vocab_size = len(preds)
                    effective_k = int(self.topk_loss.k * self.topk_loss.window_ratio)  # ← ИСПРАВЛЕНО
                    k = max(1, min(effective_k, vocab_size))

                    topk_indices = np.argpartition(preds, -k)[-k:]

                    mask = np.zeros_like(preds)
                    mask[topk_indices] = 1.0

                    preds = preds * mask
                    if np.sum(preds) > 0:
                        preds = preds / np.sum(preds)
               
                # === ДОБАВЛЕНИЕ: NUCLEUS SAMPLING (TOP-P) ===
                if hasattr(self, 'already_switched_to_topk'):
                    # Сортируем вероятности
                    sorted_indices = np.argsort(preds)[::-1]
                    sorted_probs = preds[sorted_indices]
                    cumsum = np.cumsum(sorted_probs)
                    
                    # Обрезаем до 0.95
                    cutoff_idx = np.searchsorted(cumsum, 0.95)
                    valid_indices = sorted_indices[:cutoff_idx + 1]
                    
                    # Создаем маску
                    mask = np.zeros_like(preds)
                    mask[valid_indices] = 1.0
                    
                    # Применяем маску
                    preds = preds * mask
                    if np.sum(preds) > 0:
                        preds = preds / np.sum(preds)
                

                # Предотвращаем повторение одного слова подряд
                last_idx = None
                if words and words[-1] in self.tokenizer.word_index:
                    last_idx = self.tokenizer.word_index[words[-1]]

                if last_idx is not None and last_idx < len(preds):
                    adjusted_preds = preds.copy()
                    adjusted_preds[last_idx] = 0
                    if np.sum(adjusted_preds) > 0:
                        adjusted_preds = adjusted_preds / np.sum(adjusted_preds)
                        preds = adjusted_preds

                # ===== ДОБАВЛЯЕМ ПРОВЕРКУ СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ =====
               
                next_word = None
                next_idx = None
            
                # Пробуем 10 раз выбрать случайное слово с проверкой значимости
                for attempt in range(10):
                    next_idx = np.random.choice(len(preds), p=preds)
                    candidate_word = self.tokenizer.index_word.get(next_idx, '')
                
                    if not candidate_word or candidate_word == '<OOV>':
                        continue
                
                    if self.is_word_statistically_significant(words, candidate_word):
                        next_word = candidate_word
                        break
                    else:
                        # Если незначимо - обнуляем вероятность этого слова
                        preds[next_idx] = 0
                        if np.sum(preds) > 0:
                            preds = preds / np.sum(preds)
                        else:
                            break
            
                # ПРОВЕРКА ТОП-5 НАИБОЛЕЕ ВЕРОЯТНЫХ СЛОВ
                if not next_word:
                    sorted_indices = np.argsort(preds)[::-1]
                    for idx in sorted_indices[:5]:
                        word = self.tokenizer.index_word.get(idx, '')
                        if word and word != '<OOV>' and self.is_word_statistically_significant(words, word):
                            next_word = word
                            break
            
                # ЖЁСТКИЙ FALLBACK — как предохранитель
                # ИЗМЕНЕНИЕ: Вместо тупого взятия top-1, пробуем top-3 на статистическую значимость
                if not next_word and len(sorted_indices) > 0:
                    for idx in sorted_indices[:3]: # Проверяем топ-3, а не только топ-1
                        word = self.tokenizer.index_word.get(idx, '')
                        if word and word != '<OOV>':
                            # Даже если fallback, проверяем на повторы
                            if word not in words[-2:]: 
                                next_word = word
                                break
                    if not next_word: # Если совсем все плохо
                        next_word = self.tokenizer.index_word.get(sorted_indices[0], '')
               

                if not next_word:
                    break

                words.append(next_word)

                if len(words) >= max_length:
                    break
                
            generated = ' '.join(words)
            
            # Если нужно вернуть только новые токены
            if return_only_new_tokens:
                if len(words) > len(original_words):
                    new_tokens = words[len(original_words):]
                    generated = ' '.join(new_tokens)
                else:
                    return None, 0.0  # Явный сигнал "ничего не сгенерировано"

            confidence = self.evaluate_causal_quality(generated)
            return generated, confidence
        except Exception as e:
            logging.error(f"Ошибка генерации текста: {e}")
            return seed_text, 0.5

    def compile_model(self):
        if not self.model_compiled and self.model is not None:
            
            current_weights = None
            if hasattr(self, 'model') and self.model is not None:
                try:
                    current_weights = self.model.get_weights()
                    logging.info(f"✅ Сохранены веса модели перед перекомпиляцией")
                except Exception as e:
                    logging.warning(f"Не удалось сохранить веса: {e}")
                    current_weights = None


            if self.penalty_controller is None:
                self.penalty_controller = QPenaltyController(n_penalties=4)

            from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
            lr = CosineDecayRestarts(
                initial_learning_rate=3e-4,
                first_decay_steps=800,
                t_mul=2.0,
                m_mul=0.8,
                alpha=0.05
            )

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                clipnorm=0.3,
                epsilon=1e-7
            )

            def get_adaptive_k(vocab_size):
                base_k = max(2000, int(vocab_size * 0.9))
                logging.info(f"Адаптивный K: {base_k} (словарь: {vocab_size})")
                return base_k

            
            # ИСПРАВЛЕНИЕ: временно используем обычный CE для первых эпох
            if self.initial_epoch < 50:  # Первые 50 эпох
                base_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
                logging.info(f"Используем CategoricalCrossentropy (начальная эпоха: {self.initial_epoch})")
            else:
                # Затем переключаемся на адаптивный TopK
                k_value = get_adaptive_k(self.vocab_size)
                base_loss = TopKCategoricalCrossentropy(k=k_value, window_ratio=0.05)  # Начинаем с 5%
                self.topk_loss = base_loss  # ← Сохраняем ссылку
                
                logging.info(f"Используем TopKCategoricalCrossentropy(k={k_value}, window_ratio={base_loss.window_ratio})")
                self.already_switched_to_topk = True  
  

            penalized_loss = QPenalizedCrossentropy(
                penalty_controller=self.penalty_controller,
                base_loss=base_loss
            )
            

            self.model.compile(
                optimizer=optimizer,
                loss=penalized_loss,
                metrics=['accuracy']
            )

            # ← ВСТАВЛЯЕМ ВОССТАНОВЛЕНИЕ ВЕСОВ
            if current_weights is not None:
                try:
                    self.model.set_weights(current_weights)
                    logging.info("✅ Веса восстановлены после перекомпиляции")
                except Exception as e:
                    logging.warning(f"Не удалось восстановить веса: {e}")
        
            # ← УСТАНАВЛИВАЕМ ФЛАГ ПЕРЕКОМПИЛЯЦИИ
            if hasattr(self, 'penalty_controller') and self.penalty_controller:
                self.penalty_controller.recent_recompilation = 5

            
            self.model_compiled = True
            logging.info(f"Модель скомпилирована")



    def _get_current_learning_rate(self):
        optimizer = self.model.optimizer
        if optimizer is None:
            return 0.001

        lr = optimizer.learning_rate

        if hasattr(lr, '__call__'):
            try:
                iterations = optimizer.iterations
                if hasattr(iterations, 'numpy'):
                    return float(lr(iterations).numpy())
                else:
                    return float(lr(iterations))
            except:
                pass

        if hasattr(lr, 'numpy'):
            try:
                return float(lr.numpy())
            except:
                pass

        try:
            return float(lr)
        except:
            pass

        return 0.001

    def _fallback_recompile(self):
        if self.model is None:
            self.build_model()
            self._model_built = True
        try:
            # ← ИЗМЕНЕНИЕ 4
            if self.initial_epoch < 50:
                base_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
                logging.info("Fallback: используем CategoricalCrossentropy")
            else:
                # ← ИСПРАВЛЕНИЕ: учитываем стадию обучения
                if hasattr(self, 'already_switched_to_topk') and self.already_switched_to_topk:
                    base_loss = TopKCategoricalCrossentropy(k=self.vocab_size, window_ratio=0.05)# Начинаем с 5%
                    self.topk_loss = base_loss
                    logging.info("Fallback: используем TopKCategoricalCrossentropy")
                else:
                    base_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
                    logging.info("Fallback: используем CategoricalCrossentropy")


            penalized_loss = QPenalizedCrossentropy(
                penalty_controller=self.penalty_controller,
                base_loss=base_loss
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

            self.model.compile(
                optimizer=optimizer,
                loss=penalized_loss,
                metrics=['accuracy']
            )
            self.model_compiled = True
            logging.info("Fallback перекомпиляция с адаптивным окном (50%)")
            return True
        except Exception as e:
            logging.error(f"Fallback перекомпиляция провалилась: {e}")
            return False

    def improved_emergency_escape(self, current_loss, current_epoch):
        
        # ← ДОБАВИТЬ: игнорируем первые 10 эпохи после перехода на TopK
        if hasattr(self, 'already_switched_to_topk') and self.already_switched_to_topk:
            if current_epoch < 60:  # Первые 10 эпохи после перехода
                logging.info(f"Игнорируем механизмы выхода из ямы после TopK перехода (эпоха {current_epoch})")
                return False

        if not hasattr(self, 'previous_losses') or self.previous_losses is None:
            self.previous_losses = []
        if current_loss < 3.0 or current_loss > 15.0:
            return False
        if len(self.previous_losses) < 10:
            if len(self.previous_losses) == 0:
                self.previous_losses = [current_loss] * 10
            else:
                while len(self.previous_losses) < 10:
                    self.previous_losses.append(self.previous_losses[-1])
            return False

        is_diverging = False
        if len(self.previous_losses) >= 3:
            recent_losses = [self.previous_losses[-2], self.previous_losses[-1], current_loss]
            is_consistently_increasing = (recent_losses[1] > recent_losses[0] and
                                         recent_losses[2] > recent_losses[1])
            is_diverging = is_consistently_increasing

        recent_losses = self.previous_losses[-8:]
        mean_loss = np.mean(recent_losses)
        loss_std = np.std(recent_losses)

        relative_std = loss_std / mean_loss if mean_loss > 1e-9 else 0
        is_stagnant = (relative_std < 0.02) and (mean_loss > 0.1)

        if len(recent_losses) >= 4:
            first_half_mean = np.mean(recent_losses[:4])
            second_half_mean = np.mean(recent_losses[4:])
            improvement_ratio = second_half_mean / first_half_mean if first_half_mean > 1e-9 else 1.0
            no_improvement = improvement_ratio > 0.95
        else:
            no_improvement = False

        if len(self.previous_losses) >= 12:
            long_term_first = np.mean(self.previous_losses[-12:-8])
            long_term_second = np.mean(self.previous_losses[-4:])
            long_term_ratio = long_term_second / long_term_first if long_term_first > 1e-9 else 1.0
            too_slow = long_term_ratio > 0.95
        else:
            too_slow = False

        if is_stagnant or no_improvement or too_slow or is_diverging:
            reason = []
            if is_stagnant: reason.append("стагнация")
            if no_improvement: reason.append("нет улучшений")
            if too_slow: reason.append("медленно")
            if is_diverging: reason.append("расхождение")

            logging.info(f"Обнаружена яма! Loss: {mean_loss:.4f} ({', '.join(reason)})")
            try:
                current_weights = self.model.get_weights()
                noisy_weights = []

                if is_diverging:
                    base_noise = 0.05 + 0.02 * min(1.0, current_epoch / 100.0)
                    lr_multiplier = 0.8
                else:
                    base_noise = 0.01 + 0.01 * min(1.0, current_epoch / 100.0)
                    lr_multiplier = 1.2

                import random
                if random.choice([True, False]):
                    jump_multiplier = lr_multiplier
                    strategy_name = "оригинальный"
                else:
                    jump_multiplier = lr_multiplier * 0.75
                    strategy_name = "умножение на 3/4"

                for layer_weights in current_weights:
                    if isinstance(layer_weights, np.ndarray) and layer_weights.size > 0:
                        std_val = np.std(layer_weights)
                        noise_scale = base_noise * std_val if std_val > 0 else 0.01
                        noise = np.random.normal(0, noise_scale, layer_weights.shape)
                        noisy_weights.append(layer_weights + noise)
                    else:
                        noisy_weights.append(layer_weights)
                self.model.set_weights(noisy_weights)

                current_lr = self._get_current_learning_rate()
                new_lr = current_lr * jump_multiplier
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=new_lr,
                    decay_steps=1000,
                    decay_rate=0.96,
                    staircase=True
                )
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    clipnorm=1.0
                )

                # ← ИЗМЕНЕНИЕ 5
                if self.initial_epoch <50:
                    # Используем обычный CE (как в compile_model)
                    base_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
                    logging.info(f"Emergency escape: используем CategoricalCrossentropy (initial_epoch={self.initial_epoch})")
                else:
                    # Используем TopK (как в compile_model)
                    base_loss = TopKCategoricalCrossentropy(
                        k=self.vocab_size,
                        window_ratio=0.05
                    )# Начинаем с 5%
                    logging.info(f"Emergency escape: используем TopKCategoricalCrossentropy (initial_epoch={self.initial_epoch})")
                
                self.model.compile(
                    optimizer=optimizer,
                    loss=QPenalizedCrossentropy(
                        penalty_controller=self.penalty_controller,
                        base_loss=base_loss
                    ),
                    metrics=['accuracy']
                )
                self.model_compiled = True
              

                logging.info(f"Улучшенный прыжок: LR {current_lr:.2e} to {new_lr:.2e}, шум {base_noise:.1%}, стратегия: {strategy_name}")
                if len(self.previous_losses) > 10:
                    self.previous_losses = self.previous_losses[:-4]
                return True
            except Exception as e:
                logging.error(f"Ошибка в improved_emergency_escape: {e}")
                return self._fallback_recompile()
        return False


    
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
            import keras
            keras.config.enable_unsafe_deserialization()
        
            custom_objects = {
                'MultiHeadAttentionWithHeads': MultiHeadAttentionWithHeads,
                'AttentionController': AttentionController,
                'QPenalizedCrossentropy': QPenalizedCrossentropy,
                'TopKCategoricalCrossentropy': TopKCategoricalCrossentropy,
                '_pad_mask': _pad_mask,
                '_causal_mask': _causal_mask,
                '_combine_masks': _combine_masks,
                '_square_features': _square_features,
                'tf': tf,
            }
        
            self.model = load_model(
                os.path.join(model_dir, "q_model.keras"),
                custom_objects=custom_objects
            )
            with open(os.path.join(model_dir, "tokenizer.pkl"), 'rb') as f:
                self.tokenizer = pickle.load(f)

            # Восстанавливаем penalty_controller в loss
            if hasattr(self.model, 'loss') and hasattr(self.model.loss, 'penalty_controller'):
                self.model.loss.penalty_controller = self.penalty_controller

            self.model_compiled = False
            self.compile_model()

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
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='pre')
        return padded

    def _create_fallback_training_data(self):
        basic_sequences = []
        basic_targets = []

        fallback_pairs = [
            ("текст запрос", "ответ"),
            ("вопрос модель", "обучение"),
            ("данные система", "анализ"),
            ("генерация сеть", "нейронный")
        ]

        for input_text, target_word in fallback_pairs:
            if target_word in self.tokenizer.word_index and target_word != '<OOV>':
                target_idx = self.tokenizer.word_index[target_word]
                if 0 < target_idx < self.vocab_size:
                    seq = self.texts_to_sequences([input_text])[0]
                    if np.any(seq):
                        basic_sequences.append(seq)

                        target_vec = np.zeros(self.vocab_size, dtype=np.float32)
                        target_vec[target_idx] = 0.8
                        remaining_prob = 1.0 - np.sum(target_vec)
                        if remaining_prob > 0:
                            uniform_value = remaining_prob / self.vocab_size
                            target_vec += uniform_value
                        target_vec = target_vec / np.sum(target_vec)
                        basic_targets.append(target_vec)

        return np.array(basic_sequences) if basic_sequences else None, np.array(basic_targets) if basic_targets else None

    def generate_training_chains(self, num_chains=2000):
        training_chains = []

        for pair in self.query_answer_pairs[:num_chains]:
            question = pair['question']
            answer = pair['answer']

            chain = self._create_basic_chain(question, answer)
            if chain:
                training_chains.append(chain)

            for _ in range(2):
                augmented_chain = self._create_augmented_chain(question, answer)
                if augmented_chain:
                    training_chains.append(augmented_chain)

        filtered_chains = []
        for chain in training_chains:
            if len(chain) >= 2:
                pseudo_q = chain[0]
                pseudo_a = chain[-1]
                reward = self.get_enhanced_chain_reward(pseudo_q, pseudo_a, ' '.join(chain))
                if reward > 0.3:
                    filtered_chains.append(chain)

        logging.info(f"После фильтрации осталось цепочек: {len(filtered_chains)}")
        return filtered_chains
    
        

    def _create_basic_chain(self, question, answer):
        semantic_core = self.extract_semantic_core(question, min_common_words=1)
        if not semantic_core:
            return None

        filtered_words = self.filter_relevant_words(semantic_core, question)
        if not filtered_words:
            return None

        chain = [question]
        current_text = question

        answer_words = answer.split()
        step_size = max(1, len(answer_words) // 3)

        for i in range(0, len(answer_words), step_size):
            step_words = answer_words[i:i + step_size]
            if len(step_words) > 0:
                new_step = f"{current_text} {' '.join(step_words)}"
                chain.append(new_step.strip())
                current_text = new_step

        chain.append(answer)
        return chain

    def _create_augmented_chain(self, question, answer):
        answer_words = answer.split()
        if len(answer_words) > 3:
            shuffled_words = answer_words.copy()
            random.shuffle(shuffled_words)

            chain = [question]
            current_text = question

            step_size = max(1, len(shuffled_words) // 3)
            for i in range(0, len(shuffled_words), step_size):
                step_words = shuffled_words[i:i + step_size]
                new_text = f"{current_text} {' '.join(step_words)}"
                chain.append(new_text.strip())
                current_text = new_text

            return chain
        return None

    def create_target_from_text(self, text, base_prob=0.95):
        words = text.split()[:8]

        if not words:
            return None

        correct_indices = []
        for word in words:
            if word in self.tokenizer.word_index:
                word_idx = self.tokenizer.word_index[word]
                if 0 < word_idx < self.vocab_size and word != '<OOV>':
                    correct_indices.append(word_idx)

        if not correct_indices:
            return None

        return self.create_strong_target(correct_indices, confidence=base_prob)

    def train(self, epochs=20, batch_size=64, use_early_stopping=True, loss_threshold=0.01):
        self.generation_cache = {}
        self.loss_threshold = loss_threshold

        if not hasattr(self, 'previous_losses'):
            self.previous_losses = []
        if not hasattr(self, 'penalty_controller') or self.penalty_controller is None:
            self.penalty_controller = QPenaltyController(n_penalties=4)

        # ← ИЗМЕНЕНИЕ 3: Добавляем переменные для адаптивного окна
        best_loss = float('inf')
        expansion_counter = 0
        current_window_ratio = 0.05# Начинаем с 5%

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
            # Добавляем специальный маркер для объяснений
            self.tokenizer.fit_on_texts(['[EXPLAIN]'])
            

        if self.model is None:
            loaded = self.load_from_dir(self.model_dir)
            if loaded:
                current_model_vocab = self.model.output_shape[-1]
                if current_model_vocab != self.vocab_size:
                    old_model = self.model
                    self.transfer_weights(old_model, self.vocab_size)
            else:
                self.build_model()

        state_sequences = []
        targets = []

        logging.info("Подготовка УЛУЧШЕННЫХ данных для обучения с цепочками рассуждений...")

        num_chains = min(500, len(self.query_answer_pairs))
        training_chains = self.generate_training_chains(num_chains=num_chains)

        # ДОБАВЛЕНО: Минимальный explain-объектив
        if '[EXPLAIN]' in self.tokenizer.word_index:
            explain_idx = self.tokenizer.word_index['[EXPLAIN]']
            if 0 < explain_idx < self.vocab_size:
                for pair in self.query_answer_pairs[:min(150, len(self.query_answer_pairs))]:
                    # Создаем последовательность с маркером объяснения
                    input_seq = pair['question'] + ' [EXPLAIN]'
                    seq = self.texts_to_sequences([input_seq])[0]
                    
                    if len(seq) > 1 and np.any(seq):
                        state_sequences.append(seq)
                        
                        # Цель - предсказать маркер объяснения
                        target_vector = np.zeros(self.vocab_size, dtype=np.float32)
                        target_vector[explain_idx] = 0.7
                        target_vector += (1.0 - 0.7) / self.vocab_size
                        target_vector = target_vector / np.sum(target_vector)
                        targets.append(target_vector)
                        
                        # Также обучаем обратное направление: ответ -> [EXPLAIN]
                        reverse_seq = pair['answer'] + ' [EXPLAIN]'
                        seq_rev = self.texts_to_sequences([reverse_seq])[0]
                        if len(seq_rev) > 1 and np.any(seq_rev):
                            state_sequences.append(seq_rev)
                            targets.append(target_vector)

        for chain in training_chains:
            for i in range(len(chain) - 1):
                current_step = chain[i]
                next_step = chain[i + 1]

                current_words = current_step.split()
                next_words = next_step.split()

                new_words = [word for word in next_words if word not in current_words]

                if new_words:
                    for j in range(len(new_words)):
                        state_text = current_step
                        next_word = new_words[j]

                        if next_word not in self.tokenizer.word_index or next_word == '<OOV>':
                            continue

                        word_idx = self.tokenizer.word_index[next_word]
                        if word_idx == 0 or word_idx >= self.vocab_size:
                            continue

                        state_sequence = self.texts_to_sequences([state_text])[0]

                        if not np.any(state_sequence):
                            continue

                        state_sequences.append(state_sequence)

                        target_vector = self.create_strong_target([word_idx], confidence=0.7)
                        targets.append(target_vector)

        logging.info("Добавление обучения процессу семантического ядра...")
        for pair in self.query_answer_pairs[:200]:
            question = pair['question']
            semantic_core = self.extract_semantic_core(question, min_common_words=1)
            filtered_words = self.filter_relevant_words(semantic_core, question)

            if filtered_words:
                has_valid_words = any(word in self.tokenizer.word_index and
                                      0 < self.tokenizer.word_index[word] < self.vocab_size and
                                      word != '<OOV>'
                                      for word in filtered_words)
                if not has_valid_words:
                    continue

                state_sequence = self.texts_to_sequences([question])[0]
                if np.any(state_sequence):
                    state_sequences.append(state_sequence)

                    correct_indices = []
                    for word in filtered_words:
                        if word in self.tokenizer.word_index and word != '<OOV>':
                            word_idx = self.tokenizer.word_index[word]
                            if word_idx < self.vocab_size:
                                correct_indices.append(word_idx)
                    if not correct_indices:
                        continue
                    target_vector = self.create_strong_target(correct_indices, confidence=0.7)
                    targets.append(target_vector)

        logging.info("Добавление обучения многошаговой генерации (с перекрытием и 7±2)...")
        for pair in self.query_answer_pairs[:120]:
            question = pair['question']
            answer = pair['answer']
            for variant_num in range(2):
                current_text = question
                answer_words = answer.split()

                if random.random() < 0.3:
                    random.shuffle(answer_words)

                i = 0
                while i < len(answer_words):
                    step_size = random.randint(4, 7)
                    overlap = random.randint(1, 2)

                    if i + step_size > len(answer_words):
                        step_size = len(answer_words) - i
                    if step_size < 2:
                        break

                    step_words = answer_words[i:i + step_size]
                    next_step = f"{current_text} {' '.join(step_words)}"

                    state_sequence = self.texts_to_sequences([current_text])[0]
                    if np.any(state_sequence):
                        current_words = set(current_text.split())
                        new_words = [w for w in step_words if w not in current_words]

                        correct_indices = [
                            self.tokenizer.word_index[w]
                            for w in new_words
                            if w in self.tokenizer.word_index
                            and 0 < self.tokenizer.word_index[w] < self.vocab_size
                            and w != '<OOV>'
                        ]
                        if correct_indices:
                            target_vector = self.create_strong_target(correct_indices, confidence=0.6)
                            state_sequences.append(state_sequence)
                            targets.append(target_vector)

                    current_text = next_step
                    i += step_size - overlap


        # ===== СПЕЦИАЛЬНОЕ ОБУЧЕНИЕ CORPUS_ORDER НА ПОСЛЕДОВАТЕЛЬНОСТЯХ N-ГРАММ =====
        logging.info("Обучение corpus_order на последовательностях n-грамм...")
        ngram_examples_added = 0
        seq_cache = {}
        MAX_NGRAM = 10000  # ← ДОБАВЬ ЭТУ СТРОЧКУ было 50000 потом 20000 потом 10000(на цпу)

        for seq in self.corpus_sequences[:800]:
            if len(seq) < 4:
                continue
    
            # 1. БИГРАММА → следующая БИГРАММА
            for i in range(len(seq) - 3):
                context_bigram = f"{seq[i]} {seq[i+1]}"
                target_bigram = f"{seq[i+2]} {seq[i+3]}"
        
                if context_bigram not in seq_cache:
                    seq_cache[context_bigram] = self.texts_to_sequences([context_bigram])[0]
                state_sequence = seq_cache[context_bigram]

                if np.any(state_sequence):
                    
                    start_word = seq[i+2]
                    
                    # Собираем все n-граммы, начинающиеся с start_word, С ИХ ЧАСТОТАМИ
                    ngram_candidates = {}
                    
                    # Проверяем биграммы
                    for bigram, freq in self.corpus_bigrams.items():
                        if bigram.startswith(start_word + " "):
                            ngram_candidates[bigram] = freq
                    
                    # Проверяем триграммы
                    for trigram, freq in self.corpus_trigrams.items():
                        if trigram.startswith(start_word + " "):
                            ngram_candidates[trigram] = freq
                    
                    if ngram_candidates:
                        # Выбираем ВЗВЕШЕННО по частотам
                        ngrams = list(ngram_candidates.keys())
                        freqs = list(ngram_candidates.values())
                        
                        # Нормализуем частоты в вероятности
                        total = sum(freqs)
                        if total > 0:
                            probs = [f/total for f in freqs]
                            
                            # В 70% случаев берем самую частую, в 30% - по распределению
                            if random.random() < 0.7:
                                chosen_ngram = ngrams[np.argmax(freqs)]  # самая чащая
                            else:
                                chosen_ngram = np.random.choice(ngrams, p=probs)  # по распределению
                            
                            target_words = chosen_ngram.split()
                        else:
                            target_words = [seq[i+2], seq[i+3]]
                    else:
                        target_words = [seq[i+2], seq[i+3]]
                    
                    # Создаем target_indices из ВСЕХ слов выбранной n-граммы
                    target_indices = []
                    for word in target_words:
                        if word in self.tokenizer.word_index:
                            idx = self.tokenizer.word_index[word]
                            if 0 < idx < self.vocab_size:
                                target_indices.append(idx)
                    
                    if target_indices:
                        state_sequences.append(state_sequence)
                        target_vec = self.create_strong_target(target_indices, confidence=0.9) # ,skj 0.6 
                        targets.append(target_vec)
                        ngram_examples_added += 1
                    # ← КОНЕЦ ЗАМЕНЫ

                        # ← ДОБАВЬ ЭТУ ПРОВЕРКУ
                        if ngram_examples_added >= MAX_NGRAM:
                            break  # ← Выходит только из этого внутреннего цикла
    
                if ngram_examples_added >= MAX_NGRAM:  # ← ДОБАВЬ ЭТУ ПРОВЕРКУ ТОЖЕ
                    break
            if ngram_examples_added >= MAX_NGRAM:  # ← И ЭТУ
                break
    
            # 2. ТРИГРАММА → следующее СЛОВО  
            for i in range(len(seq) - 3):
                context_trigram = f"{seq[i]} {seq[i+1]} {seq[i+2]}"
                target_word = seq[i+3]
        
                if target_word in self.tokenizer.word_index:
                    word_idx = self.tokenizer.word_index[target_word]
                    if 0 < word_idx < self.vocab_size:
                        if context_trigram not in seq_cache:
                            seq_cache[context_trigram] = self.texts_to_sequences([context_trigram])[0]
                        state_sequence = seq_cache[context_trigram]

                        if np.any(state_sequence):
                          
                            start_word = seq[i+3]
                            
                            # Собираем все n-граммы, начинающиеся с start_word, С ИХ ЧАСТОТАМИ
                            ngram_candidates = {}
                            
                            # Проверяем биграммы
                            for bigram, freq in self.corpus_bigrams.items():
                                if bigram.startswith(start_word + " "):
                                    ngram_candidates[bigram] = freq
                            
                            # Проверяем триграммы
                            for trigram, freq in self.corpus_trigrams.items():
                                if trigram.startswith(start_word + " "):
                                    ngram_candidates[trigram] = freq
                            
                            if ngram_candidates:
                                # Выбираем ВЗВЕШЕННО по частотам
                                ngrams = list(ngram_candidates.keys())
                                freqs = list(ngram_candidates.values())
                                
                                # Нормализуем частоты в вероятности
                                total = sum(freqs)
                                if total > 0:
                                    probs = [f/total for f in freqs]
                                    
                                    # В 70% случаев берем самую частую, в 30% - по распределению
                                    if random.random() < 0.7:
                                        chosen_ngram = ngrams[np.argmax(freqs)]  # самая чащая
                                    else:
                                        chosen_ngram = np.random.choice(ngrams, p=probs)  # по распределению
                                    
                                    target_words = chosen_ngram.split()
                                else:
                                    target_words = [seq[i+3], seq[i+4]] if i+4 < len(seq) else [seq[i+3]]
                            else:
                                target_words = [seq[i+3], seq[i+4]] if i+4 < len(seq) else [seq[i+3]]
                            
                            # Создаем target_indices из ВСЕХ слов выбранной n-граммы
                            target_indices = []
                            for word in target_words:
                                if word in self.tokenizer.word_index:
                                    idx = self.tokenizer.word_index[word]
                                    if 0 < idx < self.vocab_size:
                                        target_indices.append(idx)
                            
                            if target_indices:
                                state_sequences.append(state_sequence)
                                target_vec = self.create_strong_target(target_indices, confidence=0.9) # было 0,6
                                targets.append(target_vec)
                                ngram_examples_added += 1
                            # ← КОНЕЦ ЗАМЕНЫ

                            if ngram_examples_added >= MAX_NGRAM:  # ← ПРОВЕРКА И ЗДЕСЬ
                                break
                if ngram_examples_added >= MAX_NGRAM:  # ← И ЗДЕСЬ
                    break
            if ngram_examples_added >= MAX_NGRAM:  # ← И ЗДЕСЬ
                break

        logging.info(f"Добавлено {ngram_examples_added} примеров n-грамм для corpus_order")
        # ===== КОНЕЦ ОБУЧЕНИЯ CORPUS_ORDER =====

    

        logging.info("Расширенное обучение на корпусе...")
        corpus_samples = min(2000, len(self.corpus_data))
        for text in self.corpus_data[:corpus_samples]:
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
            for i in range(len(sentences) - 1):
                current_state = sentences[i]
                next_sentence = sentences[i + 1]
                if len(current_state.split()) > 1 and len(next_sentence.split()) > 1:
                    state_sequence = self.texts_to_sequences([current_state])[0]
                    if np.any(state_sequence):
                        target_vector = self.create_target_from_text(next_sentence, base_prob=0.8)
                        if target_vector is not None:
                            state_sequences.append(state_sequence)
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
                                                       padding='pre')[0]

                        state_sequences.append(state_sequence)
                        correct_indices = [self.tokenizer.word_index.get(next_word, 1)]
                        target_vector = self.create_strong_target(correct_indices, confidence=0.90)
                        targets.append(target_vector)

        if len(state_sequences) < 5:
            logging.error("Недостаточно данных для обучения")
            return None

        if targets:
            target_maxes = [np.max(t) for t in targets]
            threshold = 1e-6
            num_with_peak = sum(1 for v in target_maxes if v > threshold)
            num_empty = len(targets) - num_with_peak

            logging.info(f"DEBUG: Всего таргетов: {len(targets)}")
            logging.info(f"С явной меткой: {num_with_peak} ({num_with_peak/len(targets):.1%})")
            logging.info(f"Без метки (пропущено): {num_empty} ({num_empty/len(targets):.1%})")

            if num_with_peak > 0:
                argmaxes = [int(np.argmax(t)) for t in targets if np.max(t) > threshold]
                from collections import Counter
                c = Counter(argmaxes)
                top10 = c.most_common(10)
                logging.info(f"DEBUG: Топ-10 целевых слов: {top10}")

        # ← ВСТАВЬ СЮДА ↓↓↓
        logging.info(f"Сформировано обучающих последовательностей: {len(state_sequences)}")

        # ===== SLIDING WINDOWS (минимальная версия) =====
        if len(state_sequences) < 20000:
            additional_sequences = []
            additional_targets = []
    
            for seq, target in zip(state_sequences, targets):
                # Сдвигаем последовательность на 1-2 позиции
                for shift in [1, 2]:
                    if shift < len(seq):
                        shifted_seq = np.roll(seq, -shift)
                        # Зануляем хвост
                        shifted_seq[-shift:] = 0
                        additional_sequences.append(shifted_seq)
                        additional_targets.append(target)
    
            if additional_sequences:
                state_sequences.extend(additional_sequences)
                targets.extend(additional_targets)
                logging.info(f"Добавлено {len(additional_sequences)} сдвинутых последовательностей")
        # ===== КОНЕЦ =====

        if len(state_sequences) > 10000:
            # Берем только первые 10000 
            state_sequences = state_sequences[:10000]
            targets = targets[:10000]
            logging.info(f"Обрезано до 10000 примеров (было больше)")
        # ← КОНЕЦ ВСТАВКИ ↑↑↑

        state_sequences = np.array(state_sequences)
        targets = np.array(targets)

        logging.info(f"Размер state_sequences: {state_sequences.shape}")
        logging.info(f"Размер targets: {targets.shape}")

        target_sums = np.sum(targets, axis=1)
        logging.info(f"Проверка target векторов:")
        logging.info(f" - Min sum: {np.min(target_sums):.6f}")
        logging.info(f" - Max sum: {np.max(target_sums):.6f}")
        logging.info(f" - Mean sum: {np.mean(target_sums):.6f}")


        actual_epochs = min(epochs, 1000)
        actual_batch_size = min(128, max(16, len(state_sequences) // 2))

        logging.info(f"Параметры обучения: эпохи={actual_epochs}, batch_size={actual_batch_size}")
        logging.info(f"Начало обучения. Размер данных: {len(state_sequences)}")
        logging.info(f"История потерь: {len(self.previous_losses)} записей")

        try:
            callbacks = []
            if use_early_stopping:
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.001
                )
                reduce_lr = ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
                callbacks = [early_stopping, reduce_lr]

            if self.model is None:          # ← вот здесь
                self.build_model()
                self._model_built = True
                self.compile_model()        # ← это для новой модели

            history_losses = []

            if state_sequences.size > 0:
                w_before = self.model.get_weights()[0].flatten()[:5].copy()

            start_epoch = self.initial_epoch
            trained_epochs = 0

            for epoch in range(actual_epochs):
                current_global_epoch = start_epoch + epoch

                # ← ДОБАВЬ ЭТОТ БЛОК НАЧАЛО
                if current_global_epoch >= 50 and not getattr(self, 'already_switched_to_topk', False):
                    logging.info(f"!!! ПЕРЕКЛЮЧЕНИЕ НА TopK на эпохе {current_global_epoch} !!!")
                    self.initial_epoch = current_global_epoch  # СИНХРОНИЗАЦИЯ 50 эпох
                    self.model_compiled = False
                    self.already_switched_to_topk = True
                    self.compile_model()  # Перекомпилируем с TopK

                if epoch == 0 and not self.previous_losses:
                    current_loss = 1.0
                elif self.previous_losses:
                    current_loss = self.previous_losses[-1]
                else:
                    current_loss = 1.0

                q_agent_state = self.penalty_controller._get_state(current_loss, self.previous_losses)
                q_agent_state = np.expand_dims(q_agent_state, axis=0)
                q_agent_states = np.tile(q_agent_state, (len(state_sequences), 1))

                val_split = 0.25 if len(state_sequences) > 20 else 0.0 # ← ТОЛЬКО ЭТО 

                history_epoch = self.model.fit(
                    [state_sequences, q_agent_states],
                    targets,
                    epochs=1,
                    batch_size=actual_batch_size,
                    validation_split=val_split,  # ← ЗДЕСЬ ИСПРАВИТЬ
                    callbacks=callbacks,
                    verbose=1
                )


                if hasattr(history_epoch, 'history') and history_epoch.history and 'loss' in history_epoch.history:
                    current_loss = history_epoch.history['loss'][0]
                else:
                    predictions = self.model.predict(state_sequences, verbose=0)
                    current_loss = tf.keras.losses.categorical_crossentropy(targets, predictions).numpy().mean()



                #============ ТЕСТ-ПРОБА ГЕНЕРАЦИИ ============
                if (hasattr(self, 'already_switched_to_topk') and 
                    self.already_switched_to_topk and 
                    hasattr(self, 'corpus_data') and 
                    self.corpus_data):
    
                    for _ in range(30):  # ← ТОЛЬКО ЭТУ СТРОКУ ДОБАВИТЬ меняем количество с 10 до 30
                        try:
                            # 1. Выбираем случайный текст из корпуса
                            random_text = random.choice(self.corpus_data)
                            words = random_text.split()
            
                            if len(words) >= 4:
                                # 2. Случайное окно 3 или 4 слова
                                window_size = random.choice([3, 4])
                                max_start = len(words) - window_size
                
                                if max_start > 0:
                                    start_idx = random.randint(0, max_start)
                    
                                    # 3. Берем окно слов
                                    window_words = words[start_idx:start_idx + window_size]
                    
                                    # 4. Разделяем: prefix (первые N-1 слов), target (последнее слово)
                                    prefix_words = window_words[:-1]
                                    target_word = window_words[-1]
                    
                                    # 5. Генерация через существующий метод
                                    generated_text, _ = self.generate_text_from_seed(
                                        ' '.join(prefix_words),
                                        max_length=1,
                                        temperature=0.7
                                    )
                    
                                    # 6. Извлекаем последнее слово
                                    generated_words = generated_text.split()
                                    if generated_words:
                                        generated_word = generated_words[-1].lower()
                                        target_word_lower = target_word.lower()
                        
                                        # 7. Сверка и установка флага
                                        hit_exact = (generated_word == target_word_lower)
                                        hit_lemma = False
                                        
                                        # ИСПРАВЛЕНИЕ: проверяем наличие morph и длину результата parse
                                        if not hit_exact and hasattr(self, 'morph') and self.morph:
                                            try:
                                                gen_parsed = self.morph.parse(generated_word)
                                                target_parsed = self.morph.parse(target_word_lower)
        
                                                if gen_parsed and target_parsed:
                                                    gen_lemma = gen_parsed[0].normal_form
                                                    target_lemma = target_parsed[0].normal_form
                                                    hit_lemma = (gen_lemma == target_lemma)
                                            except Exception as e:
                                                # Лучше не глотать молча, хотя бы в debug
                                                logging.debug(f"Ошибка лемматизации: {e}")
                                                pass
                                        
                                        if hit_exact:
                                            self.penalty_controller.test_probe_hit = True
                                            self.penalty_controller.test_probe_hit_bonus = 8.0
                                            logging.info(f"✅ ТЕСТ-ПРОБА: угадано точно '{target_word}' (+8.0)")
                                        elif hit_lemma:
                                            self.penalty_controller.test_probe_hit = True
                                            self.penalty_controller.test_probe_hit_bonus = 4.0
                                            logging.info(f"🟡 ТЕСТ-ПРОБА: угадана лемма '{target_word}' (+4.0)")
                        except Exception as e:
                            pass  # Тихий пропуск ошибок


                # ← ИЗМЕНЕНИЕ 3: Адаптивное расширение окна Top-K
                if hasattr(self, 'topk_loss') and current_loss < 5.3: #предел дл СЕ - важно считаем по теории ()
                    # ВСЕГДА увеличиваем счетчик если loss < 5.3
                    expansion_counter += 1
                    
                    # Обновляем best_loss для отслеживания
                    if current_loss < best_loss:
                        best_loss = current_loss

                    if expansion_counter >= 3:  # ← БЫЛО 2
                        increment = current_window_ratio * 0.1  # ← +10% от текущего окна
                        new_ratio = min(1.0, current_window_ratio + increment)
                        self.topk_loss.update_window_ratio(new_ratio)
                        current_window_ratio = new_ratio
                        self.current_window_ratio = new_ratio  # ← ВСТАВИТЬ
                        expansion_counter = 0
                        logging.info(f"✅ Окно словаря: {new_ratio:.1%} (+{increment:.1%}, loss: {current_loss:.4f})")
                else:
                    # Если loss > 5.3 - не сбрасываем счетчик
                    expansion_counter = max(0, expansion_counter - 1) 

        
                self.current_training_loss = current_loss
                self.previous_losses.append(current_loss)
                self.penalty_controller.update_q_network(current_loss)

                logging.info(f"Эпоха {current_global_epoch + 1}: loss = {current_loss:.4f}")

                if hasattr(self, 'already_switched_to_topk') and epoch % 5 == 0:
                    try:
                        test_size = min(20, len(state_sequences))
                        hit_count = 0
                        for i in range(test_size):
                            preds = self.model.predict([state_sequences[i:i+1], q_agent_states[i:i+1]], verbose=0)[0]
            
                            # Просто берём случайные 25% от текущего адаптивного окна
                            effective_k = int(self.topk_loss.k * self.topk_loss.window_ratio)
                            top_k_adaptive = min(effective_k, self.vocab_size)
            
                            # Случайное подмножество размером 25% от адаптивного окна
                            sample_size = max(5, int(top_k_adaptive * 0.25))
            
                            sorted_indices = np.argsort(preds)[::-1]
                            top_indices = sorted_indices[:top_k_adaptive]
            
                            # Случайно выбираем 25% индексов из топ-K
                            sampled_indices = np.random.choice(top_indices, size=sample_size, replace=False)
            
                            if np.argmax(targets[i]) in sampled_indices:
                                hit_count += 1
        
                        hit_rate = hit_count / test_size
                        self.last_hit_rate = float(hit_rate)
                        logging.info(f"Top-K hit rate (25% sample): {hit_rate:.0%} (window: {self.topk_loss.window_ratio:.1%})")
                    except:
                        pass


                # ПЕРИОДИЧЕСКОЕ ОБНОВЛЕНИЕ СЛОВАРЯ И ПАТТЕРНОВ (раз в 10 эпох после 60)
                if epoch >= 60 and epoch % 10 == 0 and current_loss < 1.0:
                    try:
                        if self.query_answer_pairs:
                            random_pair = random.choice(self.query_answer_pairs[:3])
                            test_query = random_pair['question']
                            generated, confidence = self.generate_text_from_seed(
                                test_query, max_length=8, temperature=0.7
                            )
                            
                            if confidence > 0.6 and len(generated.split()) > 3:
                                # ФИЛЬТР 1: только уникальные слова, не stop-words
                                stop_words = {'и', 'в', 'на', 'с', 'по', 'о', 'от', 'до', 'не', 'но', 
                                             'из', 'у', 'за', 'же', 'а', 'то', 'это', 'что', 'как'}
                                words = [w for w in generated.split() 
                                        if w not in stop_words and len(w) > 2]
                                
                                # ФИЛЬТР 2: минимум 3 разных слова
                                unique_words = list(set(words))
                                if len(unique_words) < 3:
                                    continue
                                
                                # ФИЛЬТР 3: не добавлять, если 50% слов уже в словаре
                                existing_words = [w for w in unique_words 
                                                 if w in self.tokenizer.word_index]
                                if len(existing_words) / len(unique_words) > 0.5:
                                    continue
                                
                                # Логируем новые слова
                                new_words = [w for w in unique_words 
                                           if w not in self.tokenizer.word_index]
                                if new_words and epoch % 20 == 0:
                                    logging.info(f"Эпоха {epoch}: +{len(new_words)} новых слов из {len(unique_words)}")
                                
                                # 1. Обновляем токенизатор (только хорошие слова)
                                self.tokenizer.fit_on_texts([' '.join(unique_words)])
                                
                                # 2. Добавляем в corpus_data только если есть новые слова
                                if len(unique_words) - len(existing_words) > 1:
                                    self.corpus_data.append(generated)
                    except:
                        pass

                history_losses.append(current_loss)
                trained_epochs += 1

                escape_applied = self.improved_emergency_escape(current_loss, current_global_epoch)
                if escape_applied:
                    logging.info(f"Применен экстренный выход из ямы на эпохе {current_global_epoch}")


                # Перестройка статистики корпуса (раз в 10 эпох после 60)
                if epoch >= 60 and epoch % 10 == 0:
                    logging.info(f"Перестраиваем статистику корпуса (эпоха {epoch})")
                    self._build_corpus_chains()  # ПЕРЕСТРАИВАЕМ статистику слов/биграмм/триграмм
                    self.extract_causal_patterns()  # ИЗВЛЕКАЕМ новые паттерны


            if state_sequences.size > 0:
                w_after = self.model.get_weights()[0].flatten()[:5].copy()
                weight_change = np.mean(np.abs(w_after - w_before))
                logging.info(f"DEBUG: Среднее изменение весов: {weight_change:.6f}")
                if weight_change < 1e-8:
                    logging.warning("DEBUG: Веса не обновляются - проверьте градиенты!")

            class SimpleHistory:
                def __init__(self):
                    self.history = {'loss': history_losses}

            history = SimpleHistory()

            logging.info("Обучение улучшенной Q-сети завершено")
            self.is_trained = True

            self.initial_epoch += trained_epochs

            final_loss = history_losses[-1] if history_losses else 0.0
            logging.info(f"Финальные потери: {final_loss:.4f}")
            logging.info(f"Количество обученных эпох: {trained_epochs}")
            logging.info(f"Следующая начальная эпоха: {self.initial_epoch}")

            self.save_model(self.model_dir)
            return history
        except Exception as e:
            logging.error(f"Ошибка во время обучения: {e}")
            return None

    def find_similar_question(self, query):
        try:
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_') or len(self.tfidf_vectorizer.vocabulary_) == 0:
                logging.warning("TFIDF вектор не инициализирован")
                return None, 0.0

            if not query or not query.strip():
                return None, 0.0

            query_vector = self.tfidf_vectorizer.transform([query]).toarray()[0]

            if np.sum(query_vector) == 0:
                return None, 0.0

            similarities = []
            for pair in self.query_answer_pairs:
                question = pair['question']
                question_vector = self.tfidf_vectorizer.transform([question]).toarray()[0]

                if np.sum(query_vector) > 0 and np.sum(question_vector) > 0:
                    similarity = cosine_similarity([query_vector], [question_vector])[0][0]
                    similarities.append((pair, similarity))
                else:
                    similarities.append((pair, 0.0))

            if similarities:
                best_pair, similarity_score = max(similarities, key=lambda x: x[1])
                return best_pair, similarity_score
            return None, 0.0
        except Exception as e:
            logging.error(f"Ошибка в find_similar_question: {e}")
            return None, 0.0

    def generate_reasoning_chain(self, initial_query, max_iterations=10, max_length_per_step=5):
        current_text = initial_query
        reasoning_chain = [current_text]
        confidence = 1.0

        def topic_overlap(prev_text, new_text):
            """Вычисляет пересечение слов между двумя текстами"""
            prev_words = set(prev_text.lower().split())
            new_words = set(new_text.lower().split())
            if not prev_words:
                return 0.0
            return len(prev_words & new_words) / (len(prev_words) + 1e-9)

        STOP_WORDS = {"и", "в", "на", "что", "это", "мы", "он", "она", "они", "но", "а", "или",
                      "не", "с", "по", "к", "у", "от", "до", "из", "за", "же", "бы", "ли", "вот", "так",
                      "можно", "нужно", "стало", "было", "есть", "еще", "уже", "очень", "весь", "там", "тут"}

        def content_score(text):
            """Доля значимых слов (не стоп-слов и длиннее 2 букв)"""
            words = text.lower().split()
            if not words:
                return 0.0
            meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            return len(meaningful) / max(len(words), 1)

        def repetition_penalty(text):
            """Штраф за повторения слов"""
            words = text.lower().split()
            if len(words) < 2:
                return 1.0
            return len(set(words)) / len(words)

        for iteration in range(max_iterations):
            logging.info(f"Итерация {iteration + 1}: {current_text}")
            
            # --- БЛОК ГЕНЕРАЦИИ ВАРИАНТОВ ---
            if iteration == 0:
                semantic_core = self.extract_semantic_core(current_text, min_common_words=1)
                if not semantic_core:
                    logging.info("Не удалось извлечь семантическое ядро")
                    break
                filtered_words = self.filter_relevant_words(semantic_core, current_text)
                if not filtered_words:
                    filtered_words = semantic_core[:3]
                
                variants = self.generate_multiple_variants(
                    filtered_words,
                    num_variants=25,
                    max_length=max_length_per_step,
                    return_only_new_tokens=True
                )
            else:
                last_step_words = reasoning_chain[-1].split()
                short_context = last_step_words[-4:]
                
                variants = self.generate_multiple_variants(
                    short_context,
                    num_variants=25,
                    max_length=max_length_per_step,
                    return_only_new_tokens=True
                )
            # Критик в цикле выбора
            scored = []
            for v, conf in variants:
                if v is None:  # <-- ДОБАВИТЬ ЭТУ СТРОКУ
                    continue   # <-- И ЭТУ
                # базовая оценка
                base_score = conf
    
                # используем существующую систему критики
                temp_chain = reasoning_chain + [v]
                critic_score = self.get_enhanced_chain_reward(
                    reasoning_chain[0],     # исходный запрос
                    v,                      # кандидат как "ответ" для оценки
                    ' '.join(reasoning_chain + [v])
                )
    
                overlap_current = topic_overlap(current_text, v)
                overlap_anchor = topic_overlap(reasoning_chain[0], v)
                overlap = 0.3 * overlap_current + 0.7 * overlap_anchor
                overlap_bonus = overlap * 0.15  # Поощрение до +15% за связь с контекстом
                cs = content_score(v)
                rp = repetition_penalty(v)
                final_score = (0.5 * base_score + 0.25 * critic_score + 0.15 * cs + 0.1 * overlap) * rp
                logging.debug(f"overlap={overlap:.2f} cs={cs:.2f} rp={rp:.2f} final_score={final_score:.4f}")
                scored.append((v, final_score))

            best_variant, step_confidence = max(scored, key=lambda x: x[1])

            # ===== ПОЛИРОВКА ШАГА (ТОЛЬКО КАК ФИЛЬТР) =====
            polishing_passed = True
            polishing_reason = ""
            
            query_words = set(reasoning_chain[0].lower().split())
            variant_words = set(best_variant.lower().split())
            overlap = len(query_words.intersection(variant_words))
            overlap_ratio = overlap / (len(query_words) + 1e-9)
            
            # Отклоняем только если это тупой повтор запроса
            if overlap_ratio > 0.8 and len(variant_words) < 6:
                polishing_passed = False
                polishing_reason = f"повтор запроса (overlap={overlap}/{len(query_words)})"
                
            # Проверка на минимальную новизну
            if polishing_passed:
                existing_words = set(' '.join(reasoning_chain).lower().split())
                new_words_in_variant = variant_words - existing_words
                if len(new_words_in_variant) == 0:
                    polishing_passed = False
                    polishing_reason = "нет новых слов"
                    
            if not polishing_passed:
                logging.info(f"❌ Полировка отклонила вариант: {polishing_reason}")

            # ===== ОБНОВЛЕННАЯ ЛОГИКА ДОБАВЛЕНИЯ В ЦЕПОЧКУ =====
            if best_variant and best_variant != current_text and polishing_passed:
                reasoning_chain.append(best_variant)
                # Проверка на чрезмерное повторение слов запроса
                query_content_words = {w for w in initial_query.lower().split() if w not in STOP_WORDS and len(w) > 2}
                if query_content_words:
                    generated_text = ' '.join(reasoning_chain[1:])  # только сгенерированная часть
                    generated_words = set(generated_text.lower().split())
                    overlap_ratio = len(query_content_words & generated_words) / len(query_content_words)
                    if overlap_ratio > 0.7:
                        logging.info(f"🛑 Остановка: цепочка перенасыщена значимыми словами запроса ({overlap_ratio:.0%})")
                        break
                
                logging.info(f"➕ Добавлен новый элемент в цепочку ({len(best_variant.split())} слов)")
                confidence = min(confidence, step_confidence)
                current_text = best_variant

                # Проверка на зацикливание (с умным счетчиком)
                if len(reasoning_chain) >= 2 and best_variant == reasoning_chain[-2]:
                    if not hasattr(self, '_loop_counter'):
                        self._loop_counter = 0
                    self._loop_counter += 1
                    
                    if self._loop_counter > 2:
                        logging.info("Обнаружено зацикливание (3 повтора подряд)")
                        self._loop_counter = 0
                        break
                    else:
                        logging.info(f"⚠️ Обнаружено зацикливание (повтор {self._loop_counter}/2), пробуем продолжить")
                        # Удаляем последний добавленный элемент и продолжаем
                        reasoning_chain.pop()
                        continue
                else:
                    if hasattr(self, '_loop_counter'):
                        self._loop_counter = 0
                        
            elif not polishing_passed:
                logging.info(f"⏭️ Пропуск шага из-за непрошедшей полировки")
            else:
                logging.info("Не удалось получить вариант или вариант не изменился")
                break
            # ===== КОНЕЦ ОБНОВЛЕННОЙ ЛОГИКИ =====

        return reasoning_chain, confidence


    def generate_text(self, query, max_length=30, temperature=0.7):
        try:
            reasoning_chain, confidence = self.generate_reasoning_chain(query, max_iterations=10, max_length_per_step=15)
        
            # Путь со стрелочками
            path = " -> ".join(reasoning_chain)
            # Финальный текст
            result_text = ' '.join(reasoning_chain[1:])
            full_response = f"{path}\n\nРезультат:\n{result_text}"
        
            logging.info(f"Сгенерирована полная цепочка: {full_response}")
            logging.info(f"Уверенность: {confidence:.2f}")
        
            self.reset_generation_state()
            return full_response, confidence
    
        except Exception as e:
            logging.error(f"Ошибка в generate_text: {e}")
            self.reset_generation_state()
            return query, 0.5



    def reset_generation_state(self):
        self.generation_cache = {}
        logging.info("Состояние генерации сброшено")
