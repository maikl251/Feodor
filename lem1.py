import tkinter as tk
from tkinter import filedialog, messagebox
import re
import chardet
import json
import os
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from pymorphy2 import MorphAnalyzer
    from morph_singleton import get_morph
except ImportError:
    print("pymorphy2 не установлен. Используется упрощённая лемматизация.")
    MorphAnalyzer = None
    def get_morph():
        return None

class SemanticAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.corpus_theme = ""
        self.reference_texts = {}
        self.themes_file = "themes_data.json"
        self.load_themes()
    
    def load_themes(self):
        if os.path.exists(self.themes_file):
            try:
                with open(self.themes_file, 'r', encoding='utf-8') as f:
                    self.reference_texts = json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки тем: {e}")
    
    def save_themes(self):
        try:
            with open(self.themes_file, 'w', encoding='utf-8') as f:
                json.dump(self.reference_texts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения тем: {e}")
    
    def set_theme(self, theme):
        self.corpus_theme = theme.lower()
    
    def create_theme_from_text(self, theme_name, lemmatized_text):
        if theme_name not in self.reference_texts:
            self.reference_texts[theme_name] = [lemmatized_text]
            self.save_themes()
            return True
        return False
    
    def add_text_to_theme(self, lemmatized_text):
        if self.corpus_theme in self.reference_texts:
            self.reference_texts[self.corpus_theme].append(lemmatized_text)
            if len(self.reference_texts[self.corpus_theme]) > 10:
                self.reference_texts[self.corpus_theme] = self.reference_texts[self.corpus_theme][-10:]
            self.save_themes()
            return True
        return False
    
    def analyze_semantics(self, lemmatized_text):
        if not self.corpus_theme:
            return {"error": "Тема корпуса не установлена"}
        
        reference_docs = self.reference_texts.get(self.corpus_theme, [])
        if not reference_docs:
            return {"error": f"Нет эталонных текстов для темы '{self.corpus_theme}'"}
        
        all_docs = reference_docs + [lemmatized_text]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_docs)
            query_vector = tfidf_matrix[-1]
            reference_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vector, reference_vectors)
            avg_similarity = np.mean(similarities)
            
            words = lemmatized_text.split()
            word_freq = Counter(words)
            top_keywords = word_freq.most_common(10)
            
            relevance = "высокая" if avg_similarity > 0.3 else "средняя" if avg_similarity > 0.1 else "низкая"
            
            return {
                "theme": self.corpus_theme,
                "similarity_score": round(float(avg_similarity), 3),
                "relevance": relevance,
                "top_keywords": [{"word": word, "count": count} for word, count in top_keywords],
                "total_words": len(words),
                "unique_words": len(word_freq)
            }
        except Exception as e:
            return {"error": f"Ошибка анализа: {str(e)}"}
    
    def create_tfidf_vectors(self, lemmatized_text):
        try:
            words = lemmatized_text.split()
            word_freq = Counter(words)
            all_words = set()
            for theme_texts in self.reference_texts.values():
                for text in theme_texts:
                    all_words.update(text.split())
            all_words.update(words)
            
            vectorizer = TfidfVectorizer(vocabulary=list(all_words))
            all_docs = []
            for theme_texts in self.reference_texts.values():
                all_docs.extend(theme_texts)
            all_docs.append(lemmatized_text)
            
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            current_vector = tfidf_matrix[-1].toarray()[0]
            
            keyword_vectors = []
            feature_names = vectorizer.get_feature_names_out()
            for word, count in word_freq.most_common(10):
                if word in feature_names:
                    word_index = np.where(feature_names == word)[0][0]
                    tfidf_score = current_vector[word_index]
                else:
                    tfidf_score = 0.0
                keyword_vectors.append({
                    "word": word,
                    "count": count,
                    "tfidf_score": round(float(tfidf_score), 4)
                })
            return keyword_vectors
        except Exception as e:
            print(f"Ошибка создания TF-IDF векторов: {e}")
            return []

class LemmatizerProcessor:
    def __init__(self):
        self.morph = get_morph()
        self.semantic_analyzer = SemanticAnalyzer()
        self.text_content = ""
        self.lemmatized_text = ""
        self.analysis_results = {}
    
    def process_file(self, file_path, theme_name):
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if encoding is None or confidence < 0.7:
                encodings = ['utf-8', 'cp1251', 'iso-8859-1', 'koi8-r']
                for enc in encodings:
                    try:
                        self.text_content = raw_data.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError("Не удалось определить кодировку файла")
            else:
                self.text_content = raw_data.decode(encoding)
            
            words = re.findall(r'\b\w+\b', self.text_content.lower())
            if self.morph:
                self.lemmatized_text = ' '.join(self.morph.parse(word)[0].normal_form for word in words)
            else:
                self.lemmatized_text = ' '.join(words)
            
            self.semantic_analyzer.set_theme(theme_name)
            if theme_name not in self.semantic_analyzer.reference_texts:
                self.semantic_analyzer.create_theme_from_text(theme_name, self.lemmatized_text)
            else:
                self.semantic_analyzer.add_text_to_theme(self.lemmatized_text)
            
            self.analysis_results = self.semantic_analyzer.analyze_semantics(self.lemmatized_text)
            
            self.add_to_base_corpus(self.lemmatized_text)
            
            return self.lemmatized_text
        except Exception as e:
            print(f"Ошибка обработки файла {file_path}: {e}")
            return None
    
    def add_to_base_corpus(self, lemmatized_text):
        base_corpus_file = "base_corpus.json"
        if os.path.exists(base_corpus_file):
            with open(base_corpus_file, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
        else:
            base_data = []
        base_data = list(set(base_data + [lemmatized_text]))
        with open(base_corpus_file, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
    
    def save_for_vae_training(self, output_folder, file_number):
        try:
            corpus_data = {
                "file_number": file_number,
                "original_text": self.text_content,
                "lemmatized_text": self.lemmatized_text,
                "tokens": self.lemmatized_text.split(),
                "token_count": len(self.lemmatized_text.split())
            }
            corpus_filename = os.path.join(output_folder, f"корпус{file_number}.json")
            with open(corpus_filename, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=4)
            
            if "error" not in self.analysis_results:
                keyword_vectors = self.semantic_analyzer.create_tfidf_vectors(self.lemmatized_text)
                semantic_data = {
                    "file_number": file_number,
                    "theme": self.analysis_results['theme'],
                    "similarity_score": self.analysis_results['similarity_score'],
                    "relevance": self.analysis_results['relevance'],
                    "total_words": self.analysis_results['total_words'],
                    "unique_words": self.analysis_results['unique_words'],
                    "top_keywords": self.analysis_results['top_keywords'],
                    "keyword_vectors": keyword_vectors
                }
                semantic_filename = os.path.join(output_folder, f"семантика{file_number}.json")
                with open(semantic_filename, 'w', encoding='utf-8') as f:
                    json.dump(semantic_data, f, ensure_ascii=False, indent=4)
            else:
                semantic_data = {
                    "file_number": file_number,
                    "error": self.analysis_results['error']
                }
                semantic_filename = os.path.join(output_folder, f"семантика{file_number}.json")
                with open(semantic_filename, 'w', encoding='utf-8') as f:
                    json.dump(semantic_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Ошибка сохранения для VAE: {e}")
