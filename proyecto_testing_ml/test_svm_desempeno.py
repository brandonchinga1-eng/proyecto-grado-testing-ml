"""
test_svm_desempeno.py
------------------------------------------------------------
Suite de pruebas que VALIDA el desempeño del modelo SVM
seleccionado como modelo definitivo del prototipo.

Estas pruebas verifican que las metricas reportadas en el
documento (exactitud, kappa, validacion cruzada) son reales
y reproducibles.

Uso:  pytest test_svm_desempeno.py -v
------------------------------------------------------------
"""
import pytest
import numpy as np
import math
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier

# Cargar dataset desde evaluacion_modelo.py
exec(open("evaluacion_modelo.py").read().split(
    "# =========================\n# PROCESAMIENTO")[0])


@pytest.fixture(scope="module")
def datos():
    """Prepara el dataset vectorizado (compartido por todas las pruebas)."""
    vec = TfidfVectorizer()
    X = vec.fit_transform(textos)
    y = np.array(etiquetas)
    return X, y


@pytest.fixture(scope="module")
def split_80_20(datos):
    """Particion 80/20 estandar reproducible."""
    X, y = datos
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def modelo_svm(split_80_20):
    """Modelo SVM entrenado con la particion estandar."""
    Xtr, Xte, ytr, yte = split_80_20
    m = SVC(kernel='linear', random_state=42)
    m.fit(Xtr, ytr)
    return m


# ============================================================
# Pruebas que validan el desempeño del SVM como modelo final
# ============================================================

def test_exactitud_minima_particion_unica(modelo_svm, split_80_20):
    """El SVM debe alcanzar al menos 60% de exactitud en particion 80/20."""
    Xtr, Xte, ytr, yte = split_80_20
    acc = accuracy_score(yte, modelo_svm.predict(Xte))
    assert acc >= 0.60, f"Exactitud {acc:.2%} por debajo del minimo esperado (60%)"


def test_exactitud_supera_lineas_base(modelo_svm, split_80_20):
    """El SVM debe superar claramente las lineas base aleatorias."""
    Xtr, Xte, ytr, yte = split_80_20
    acc_modelo = accuracy_score(yte, modelo_svm.predict(Xte))
    # Linea base: clase mas frecuente
    d = DummyClassifier(strategy="most_frequent")
    d.fit(Xtr, ytr)
    acc_base = accuracy_score(yte, d.predict(Xte))
    assert acc_modelo > acc_base * 3, \
        f"El modelo ({acc_modelo:.2%}) no supera 3x la linea base ({acc_base:.2%})"


def test_kappa_minimo(modelo_svm, split_80_20):
    """El kappa de Cohen debe indicar al menos acuerdo moderado (>= 0.4)."""
    Xtr, Xte, ytr, yte = split_80_20
    k = cohen_kappa_score(yte, modelo_svm.predict(Xte))
    assert k >= 0.4, f"Kappa {k:.3f} por debajo del minimo (0.4, acuerdo moderado)"


def test_validacion_cruzada_estable(datos):
    """La exactitud media en validacion cruzada k=5 debe ser >= 75%."""
    X, y = datos
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(SVC(kernel='linear', random_state=42),
                             X, y, cv=skf, scoring='accuracy')
    media = scores.mean()
    assert media >= 0.75, \
        f"Exactitud media en CV ({media:.2%}) por debajo del minimo (75%)"


def test_svm_supera_arbol_decision(datos):
    """Verifica empiricamente que SVM > Arbol de Decision en este dataset."""
    from sklearn.tree import DecisionTreeClassifier
    X, y = datos
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_scores = cross_val_score(SVC(kernel='linear', random_state=42),
                                 X, y, cv=skf, scoring='accuracy')
    arbol_scores = cross_val_score(DecisionTreeClassifier(max_depth=10, random_state=42),
                                   X, y, cv=skf, scoring='accuracy')
    assert svm_scores.mean() > arbol_scores.mean(), \
        f"SVM ({svm_scores.mean():.2%}) no supera al Arbol ({arbol_scores.mean():.2%})"


def test_intervalo_confianza_supera_azar(datos):
    """El limite inferior del IC 95% debe estar por encima de la linea base aleatoria."""
    X, y = datos
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    s = cross_val_score(SVC(kernel='linear', random_state=42),
                        X, y, cv=skf, scoring='accuracy')
    media, desv, n = s.mean(), s.std(), len(s)
    err = desv / math.sqrt(n)
    ic_low = media - 2.262 * err  # t=2.262 para 9 gl
    azar = 1 / 7  # 7 categorias = 14.3%
    assert ic_low > azar, \
        f"IC inferior ({ic_low:.2%}) no supera el azar ({azar:.2%})"
