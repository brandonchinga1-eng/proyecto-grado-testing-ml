"""
comparar_modelos.py
------------------------------------------------------------
Compara experimentalmente CINCO tecnicas de clasificacion
sobre el mismo dataset y bajo identicas condiciones.

Genera la evidencia que respalda la Tabla 17 del documento
(seleccion del modelo definitivo).

Uso:  python comparar_modelos.py
------------------------------------------------------------
"""
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Reutiliza el dataset definido en evaluacion_modelo.py
exec(open("evaluacion_modelo.py").read().split(
    "# =========================\n# PROCESAMIENTO")[0])

print("="*65)
print("COMPARACION EXPERIMENTAL DE TECNICAS DE CLASIFICACION")
print("="*65)
print(f"Dataset: {len(textos)} muestras, {len(set(etiquetas))} categorias")
print(f"Vectorizacion: TF-IDF")
print(f"Validacion: cruzada estratificada (k=5, random_state=42)")
print("-"*65)

vec = TfidfVectorizer()
X = vec.fit_transform(textos)
y = np.array(etiquetas)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    "SVM (kernel lineal)":      SVC(kernel='linear', random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes":              MultinomialNB(),
    "K-Vecinos (KNN)":          KNeighborsClassifier(n_neighbors=3),
    "Arbol de Decision (max_depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
}

resultados = []
for nombre, modelo in modelos.items():
    s = cross_val_score(modelo, X, y, cv=skf, scoring='accuracy')
    resultados.append((nombre, s.mean(), s.std()))

# Ordenar de mayor a menor exactitud
resultados.sort(key=lambda r: -r[1])

print(f"\n{'Posicion':<10}{'Tecnica':<38}{'Exactitud':<12}{'Desv.est.'}")
print("-"*65)
for pos, (nombre, media, desv) in enumerate(resultados, 1):
    print(f"{pos}°{'':<7}{nombre:<38}{media*100:>6.1f}%{'':<5}{desv*100:>5.1f}%")

print("-"*65)
print(f"\nCONCLUSION:")
print(f"  La tecnica con mejor desempeño es '{resultados[0][0]}' con {resultados[0][1]*100:.1f}%.")
print(f"  Esta tecnica se selecciona como modelo definitivo del prototipo.")
