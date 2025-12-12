# ---------------------------
# DETECCIÓN DE IRONÍA (TF-IDF + Naive Bayes)
# Código listo para ejecutar
# ---------------------------

# 0) Librerías
import pandas as pd                                 # manejo de datos tabulares
import numpy as np                                  # operaciones numéricas
import re                                            # expresiones regulares para limpiar texto
import nltk                                          # para stopwords y procesamiento básico
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer      # stemmer para español
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Descargar stopwords si hace falta (se ejecuta solo la primera vez)
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# 1) Cargar dataset (archivo Excel desde el repo)
url = "https://github.com/aperezn298/CienciaDatosSENA/raw/main/04Datasets/dataset_ironia_sarcasmo.xlsx"
df = pd.read_excel(url)

# 2) Exploración rápida (opcional, útil para sustentar)
print("Filas, columnas:", df.shape)
print(df.head())

# 3) Preprocesamiento de texto: limpieza básica + tokenización simple + stemming
spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")

def clean_text(text):
    """
    - Pasa a minúsculas
    - Quita URLs, menciones, caracteres no alfabéticos
    - Elimina espacios extra
    - Retorna string "limpio"
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # eliminar URLs
    text = re.sub(r'@\w+', '', text)                      # eliminar menciones
    text = re.sub(r'[^a-záéíóúñü\s]', ' ', text)          # mantener letras y espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_stem(text):
    """
    - Limpia el texto
    - Separa por espacios
    - Remueve stopwords
    - Aplica stemming
    - Devuelve lista de tokens (strings)
    """
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in spanish_stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# Aplicar limpieza (crea columna auxiliar)
df['texto_limpio'] = df['texto'].astype(str).apply(clean_text)

# 4) Manejo de nulos en etiquetas (si hay)
# Aseguramos que 'es_ironico' exista y sea entero 0/1
df = df.dropna(subset=['es_ironico'])                    # quitamos filas sin etiqueta
df['es_ironico'] = df['es_ironico'].astype(int)

# 5) Preparar features con TF-IDF (usar tokenizer personalizado)
tfidf = TfidfVectorizer(
    tokenizer=lambda text: tokenize_and_stem(text),   # tokenizador definido arriba
    ngram_range=(1,2),                                # unigrams + bigrams para captar patrones
    min_df=2,                                         # ignorar tokens muy raros
    max_df=0.9                                        # ignorar tokens extremadamente frecuentes
)

X = tfidf.fit_transform(df['texto_limpio'])          # matriz TF-IDF
y = df['es_ironico']                                  # etiqueta objetivo

# 6) División entrenamiento / prueba
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.30, random_state=42, stratify=y
)

# 7) Entrenar Multinomial Naive Bayes (modelo clásico y explicable)
model = MultinomialNB(alpha=1.0)   # alpha=1 -> Laplace smoothing
model.fit(X_train, y_train)

# 8) Evaluación básica
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]   # probabilidad de clase 1 (irónico)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== Métricas del modelo Naive Bayes (TF-IDF) ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print("Matriz de confusión:\n", cm)
print("\nReporte por clases:\n", classification_report(y_test, y_pred, target_names=['no_ironico','ironico']))

# 9) ¿Cuántos casos fueron clasificados incorrectamente?
n_mal = np.sum(y_test.values != y_pred)
print(f"Casos mal clasificados en el set de prueba: {n_mal} de {len(y_test)} ({n_mal/len(y_test):.2%})")

# Mostrar ejemplo de errores para sustentar (5 primeros)
errors = df.loc[idx_test[y_test.values != y_pred]]
errors = errors.copy()
errors['pred'] = y_pred[y_test.values != y_pred]
errors['prob_pred'] = y_proba[y_test.values != y_pred]
print("\nEjemplos de errores (primeras 5 filas):")
print(errors[['texto', 'texto_limpio', 'es_ironico', 'pred', 'prob_pred']].head(5))

# 10) Evaluación de cómo falla un modelo tradicional de análisis de sentimiento
# Definimos un lexicon muy simple de palabras positivas/negativas (ejemplo didáctico)
positivas = {"excelent", "maravill", "genial", "bueno", "buena", "encant", "agrad"}
negativas = {"horribl", "mal", "pésim", "pésim", "terribl", "espera", "tard", "caro", "robo", "peor", "malo"}

def lexicon_sentiment(text):
    """
    Score simple: +1 por palabra positiva, -1 por palabra negativa.
    Retorna 'positivo', 'negativo' o 'neutro' según suma.
    (Este es un ejemplo de 'modelo tradicional' muy básico)
    """
    tokens = tokenize_and_stem(text)
    score = 0
    for t in tokens:
        if t in positivas:
            score += 1
        if t in negativas:
            score -= 1
    if score > 0:
        return "positivo"
    elif score < 0:
        return "negativo"
    else:
        return "neutro"

# Aplicar el lexicon al conjunto de prueba (texto original para interpretación)
subset = df.loc[idx_test].copy()
subset['lex_sent'] = subset['texto'].apply(lexicon_sentiment)

# Comparar lexicon vs etiqueta 'sentimiento_real' (si existe)
if 'sentimiento_real' in subset.columns:
    # número de casos donde lexicon falla respecto al sentimiento real
    mask_lexicon_error = subset['lex_sent'] != subset['sentimiento_real']
    n_lexicon_error = mask_lexicon_error.sum()
    print(f"\nCasos donde el lexicon simple difiere del 'sentimiento_real': {n_lexicon_error} de {len(subset)}")
    # cuántos de esos son ironía (es_ironico == 1)
    n_lexicon_error_iron = subset[mask_lexicon_error & (subset['es_ironico']==1)].shape[0]
    print(f"De esos, {n_lexicon_error_iron} son frases irónicas -> evidencia de fallo por ironía")
else:
    print("\nLa columna 'sentimiento_real' no está en el subconjunto de prueba; no se pudo comparar lexicon vs etiqueta real.")

# 11) Señales humanas y estrategias (impreso para la entrega)
explicacion = """
Señales humanas que ayudan a detectar ironía/sarcasmo:
- Contraste entre palabras positivas y contexto negativo (ej.: "¡qué excelente servicio, me dejaron esperando 2 horas!").
- Uso de puntuación (puntos suspensivos, mayúsculas con intención, comillas).
- Exageración o hipérbole ("me encanta pagar más por lo mismo").
- Indicios pragmáticos: sarcasmo dirigido a una entidad conocida (restaurante, servicio).
- Conocimiento del mundo: saber que algo es objetivamente malo.
    
Estrategias para mejorar la clasificación:
- Usar embeddings/contextualizados (transformers en español: BETO, BERT multilingual, etc.).
- Añadir features pragmáticos: puntuación, emoji, uso de mayúsculas, presencia de contraste léxico.
- Modelos que consideren contexto (historial del autor o conversación).
- Aumentar dataset con ejemplos anotados específicamente de ironía y realizar data aumentation.
"""
print(explicacion)

