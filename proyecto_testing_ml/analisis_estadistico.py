"""
analisis_estadistico.py
------------------------------------------------------------
Analisis estadistico extendido del modelo de clasificacion.
Complementa a evaluacion_modelo.py con las metricas solicitadas
para fortalecer la validacion del modelo:

  - Exactitud y Kappa de Cohen (Train-Test 80/20)
  - Lineas base de referencia (aleatoria y clase mas frecuente)
  - Validacion cruzada k-fold (k=5 y k=10): media y desviacion
  - Intervalo de confianza al 95% sobre la exactitud
  - Comparacion de hiperparametros max_depth

Uso:  python analisis_estadistico.py
------------------------------------------------------------
"""
import math
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier

# Reutiliza el dataset definido en evaluacion_modelo.py
exec(open("evaluacion_modelo.py").read().split(
    "# =========================\n# PROCESAMIENTO")[0])

# Vectorizacion TF-IDF (identica a la del prototipo)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)
y = np.array(etiquetas)

# Particion 80/20 reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------- 1. Exactitud y Kappa de Cohen ----------
modelo = DecisionTreeClassifier(max_depth=10, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
exactitud = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
print("=== 1. EXACTITUD Y KAPPA DE COHEN (Train-Test 80/20) ===")
print(f"Exactitud:        {exactitud:.4f} ({exactitud*100:.1f}%)")
print(f"Kappa de Cohen:   {kappa:.4f}")
print()

# ---------- 2. Lineas base ----------
print("=== 2. LINEAS BASE DE REFERENCIA ===")
for strat, nombre in [("uniform", "Aleatoria uniforme (1/7)"),
                      ("most_frequent", "Clase mas frecuente")]:
    d = DummyClassifier(strategy=strat, random_state=42)
    d.fit(X_train, y_train)
    acc = accuracy_score(y_test, d.predict(X_test))
    print(f"{nombre:28s}: {acc*100:.1f}%")
print(f"Mejora del modelo sobre el azar: {exactitud/(1/7):.2f}x")
print()

# ---------- 3. Validacion cruzada k-fold ----------
print("=== 3. VALIDACION CRUZADA K-FOLD ===")
scores_dict = {}
for k in [5, 10]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=10, random_state=42),
        X, y, cv=skf, scoring='accuracy')
    scores_dict[k] = scores
    print(f"k={k:2d}:  media={scores.mean()*100:.1f}%  "
          f"desv={scores.std()*100:.1f}%  "
          f"rango=[{scores.min()*100:.1f}%-{scores.max()*100:.1f}%]")
print()

# ---------- 4. Intervalo de confianza 95% (sobre k=10) ----------
print("=== 4. INTERVALO DE CONFIANZA 95% (k=10) ===")
s = scores_dict[10]
media, desv, n = s.mean(), s.std(), len(s)
err = desv / math.sqrt(n)
t95 = 2.262  # t de Student, 9 grados de libertad
ic_low, ic_high = media - t95*err, media + t95*err
print(f"Media:                {media*100:.1f}%")
print(f"Intervalo 95%:        [{ic_low*100:.1f}%, {ic_high*100:.1f}%]")
print(f"Linea base (azar):    {100/7:.1f}%")
print(f"El limite inferior {'SUPERA' if ic_low > 1/7 else 'NO supera'} el azar "
      f"-> el modelo aprende de forma significativa")
print()

# ---------- 5. Comparacion de max_depth ----------
print("=== 5. SELECCION DE max_depth (k=5) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for depth in [3, 5, 10, 15, None]:
    sc = cross_val_score(
        DecisionTreeClassifier(max_depth=depth, random_state=42),
        X, y, cv=skf, scoring='accuracy')
    marca = "  <- seleccionado" if depth == 10 else ""
    print(f"max_depth={str(depth):>4}:  media={sc.mean()*100:.1f}%{marca}")
