# Proyecto de Grado — Testing de Software con Machine Learning

Prototipo funcional que integra técnicas de Machine Learning (SVM con kernel lineal + TF-IDF) con automatización de pruebas (Selenium + Pytest) sobre una aplicación web Django.

## Autor

Brandon Steven Ibáñez Archila — Universidad de Investigación y Desarrollo (UDI)

## Modelo de clasificación

El modelo definitivo del prototipo es una **Máquina de Vectores de Soporte (SVM)** con kernel lineal. Esta técnica fue seleccionada tras una comparación experimental de cinco algoritmos (SVM, Random Forest, Naive Bayes, KNN y Árbol de Decisión) sobre el mismo dataset, en la que el SVM obtuvo el mejor desempeño (82.2% bajo validación cruzada k=5, frente a 48.9% del Árbol de Decisión). El análisis del Árbol de Decisión se conserva como parte del proceso investigativo que fundamenta la selección.

## Estructura

- `django-pools-master/` — Aplicación web Django usada como sistema bajo prueba
- `proyecto_testing_ml/` — Módulos de ML, predictor, análisis y pruebas automatizadas
  - `evaluacion_modelo.py` — Entrenamiento y evaluación del clasificador SVM (modelo definitivo)
  - `fase1_arbol_decision.py` — Primera fase: entrenamiento y evaluación del Árbol de Decisión (modelo inicial)
  - `predictor.py` — Carga del modelo SVM serializado
  - `comparar_modelos.py` — Comparación experimental de 5 técnicas de clasificación
  - `analisis_estadistico.py` — Kappa de Cohen, líneas base, validación cruzada e intervalo de confianza
  - `test_model.py` — Pruebas unitarias del clasificador (8 tests)
  - `test_svm_desempeno.py` — Pruebas que validan el desempeño del modelo SVM (6 tests)
  - `test_inteligente.py` — Pruebas de integración ML + Selenium (7 tests)
  - `test_selenium.py` — Pruebas de interfaz web (3 tests)

## Ejecución

### 1. Entrenar el modelo

```bash
cd proyecto_testing_ml
python evaluacion_modelo.py
```

Esto genera `modelo_svm.pkl` y `vectorizer.pkl`.

### 2. Ejecutar la comparación de técnicas

```bash
python comparar_modelos.py
```

### 3. Ejecutar el análisis estadístico

```bash
python analisis_estadistico.py
```

### 4. Ejecutar las pruebas

Levantar el servidor Django en una terminal:

```bash
cd django-pools-master/django-pools-master/django-poll
python manage.py runserver
```

En otra terminal, ejecutar las pruebas:

```bash
cd proyecto_testing_ml
pytest -v
```

## Resultados

- Dataset: 90 muestras, 7 categorías
- Modelo seleccionado: SVM (kernel lineal)
- Exactitud del modelo: 61% (partición única) / 82.2% (validación cruzada k=5)
- Kappa de Cohen: 0.55 (acuerdo moderado)
- Intervalo de confianza 95%: [79.3%, 91.8%]
- Suite de pruebas: 24 tests (8 unitarios + 6 de desempeño SVM + 7 de integración + 3 de interfaz). Los 14 primeros no requieren servidor; los 10 de integración/interfaz requieren Django activo y Chrome.
