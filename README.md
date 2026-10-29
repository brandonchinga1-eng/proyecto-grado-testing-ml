# Proyecto de Grado — Testing de Software con Machine Learning

Prototipo funcional que integra Machine Learning (**SVM con kernel lineal + TF-IDF**) con automatización de pruebas (Selenium + Pytest) sobre la aplicación web Django-Polls.

**Autor:** Brandon Steven Ibáñez Archila — Universidad de Investigación y Desarrollo (UDI)

## Modelo de clasificación

El proyecto partió de un **Árbol de Decisión** (primera fase). Tras evaluarlo (exactitud 38.9 % en partición única; 50.0 % en validación cruzada k=10) se comparó con otras cuatro técnicas sobre el mismo dataset, y las **Máquinas de Vectores de Soporte (SVM)** obtuvieron el mejor desempeño (82.2 % frente a 48.9 % del árbol, k=5). El SVM es el modelo definitivo del prototipo.

## Estructura

- `proyecto_testing_ml/` — modelo, scripts de análisis y pruebas
  - `evaluacion_modelo.py` — entrena y evalúa el modelo definitivo (SVM); contiene el dataset
  - `predictor.py` — carga el modelo serializado
  - `fase1_arbol_decision.py` — primera fase: Árbol de Decisión
  - `comparar_modelos.py` — comparación de cinco técnicas
  - `analisis_estadistico.py` — kappa, líneas base, validación cruzada, intervalo de confianza
  - `test_model.py` (8), `test_inteligente.py` (7), `test_selenium.py` (3) — pruebas funcionales
  - `test_svm_desempeno.py` (6) — pruebas que validan el desempeño del SVM
- `django-pools-master/` — aplicación Django-Polls (sistema bajo prueba), en `django-pools-master/django-pools-master/django-poll`

## Requisitos: dos entornos de Python

La aplicación usa **Django 2.2.24**, que no funciona con Python reciente.

| Entorno | Python | Instalación |
|---|---|---|
| (a) Modelo y pruebas | 3.10 o superior | `pip install scikit-learn numpy joblib pytest selenium` |
| (b) Aplicación Django | 3.9 | `pip install Django==2.2.24 pytz==2019.1 sqlparse==0.3.0` |

Se necesita además Google Chrome y Git.

## Ejecución (Windows)

```
git clone https://github.com/brandonchinga1-eng/proyecto-grado-testing-ml
cd proyecto-grado-testing-ml
py -m venv venv
venv\Scripts\python.exe -m pip install scikit-learn numpy joblib pytest selenium
py -3.9 -m venv venv_django
venv_django\Scripts\python.exe -m pip install Django==2.2.24 pytz==2019.1 sqlparse==0.3.0
```

**Sin servidor** (desde `proyecto_testing_ml`):
```
..\venv\Scripts\python.exe evaluacion_modelo.py
..\venv\Scripts\python.exe comparar_modelos.py
..\venv\Scripts\python.exe analisis_estadistico.py
..\venv\Scripts\python.exe -m pytest test_svm_desempeno.py -v
```

**Con servidor** — terminal 1 (dejarla abierta):
```
cd django-pools-master\django-pools-master\django-poll
..\..\..\venv_django\Scripts\python.exe manage.py migrate
..\..\..\venv_django\Scripts\python.exe manage.py runserver
```
Terminal 2:
```
cd proyecto_testing_ml
..\venv\Scripts\python.exe -m pytest test_model.py test_inteligente.py test_selenium.py -v
```

## Resultados

- Dataset: 90 muestras, 7 categorías (redactado por un único autor; ver limitaciones en el informe)
- SVM: 61.1 % en partición única (18 muestras); 82.2 % (k=5) y 85.6 % (k=10); kappa de Cohen 0.55; IC 95 % [79.3 %, 91.8 %]
- Árbol de Decisión (fase inicial): 38.9 % en partición única; 50.0 % (k=10)
- Pruebas: 18 de 18 aprobadas (42.60 s) y 6 de 6 de desempeño del SVM
