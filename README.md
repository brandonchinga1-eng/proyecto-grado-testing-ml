# Proyecto de Grado — Testing de Software con Machine Learning

Prototipo funcional que integra técnicas de Machine Learning (Árbol de Decisión + TF-IDF) con automatización de pruebas (Selenium + Pytest) sobre una aplicación web Django.

## Autor

Brandon Steven Ibáñez Archila — Universidad de Investigación y Desarrollo (UDI)

**Director:** MSc (F) Martín Pérez Jaimes

**Línea de investigación:** Ingeniería de Software y Aplicaciones de Inteligencia Artificial

## Descripción

Este proyecto demuestra la viabilidad técnica de integrar un clasificador de aprendizaje automático en un pipeline de pruebas de software automatizadas. A partir de la descripción textual de un caso de prueba, el sistema predice su categoría y ejecuta la prueba correspondiente sobre una aplicación web real mediante Selenium WebDriver.

## Estructura del proyecto

```
proyecto-grado-testing-ml/
├── django-pools-master/         # Aplicación web Django (sistema bajo prueba)
│   └── django-pools-master/
│       └── django-poll/
│           ├── manage.py
│           └── polls/           # App de encuestas con tests.py
│
├── proyecto_testing_ml/         # Módulos de ML y pruebas automatizadas
│   ├── evaluacion_modelo.py     # Entrenamiento y evaluación del clasificador
│   ├── predictor.py             # Carga del modelo serializado
│   ├── test_model.py            # Pruebas unitarias del clasificador (8 tests)
│   ├── test_inteligente.py      # Pruebas de integración ML + Selenium (7 tests)
│   ├── test_selenium.py         # Pruebas de interfaz web (3 tests)
│   ├── modelo_decision_tree.pkl # Modelo entrenado serializado
│   └── vectorizer.pkl           # Vectorizador TF-IDF serializado
│
├── .gitignore
└── README.md
```

## Tecnologías utilizadas

- **Python 3.14**
- **Scikit-learn** — Árbol de Decisión y vectorización TF-IDF
- **Pandas** — Manipulación del dataset
- **Selenium WebDriver** — Automatización de navegador
- **Pytest** — Framework de pruebas
- **Django** — Aplicación web bajo prueba
- **Joblib** — Serialización del modelo entrenado

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/brandonchinga1-eng/proyecto-grado-testing-ml.git
cd proyecto-grado-testing-ml
```

2. Instalar dependencias:

```bash
pip install scikit-learn pandas selenium pytest django joblib
```

## Ejecución

### 1. Levantar el servidor Django

En una terminal:

```bash
cd django-pools-master/django-pools-master/django-poll
python manage.py migrate
python manage.py runserver
```

El servidor estará disponible en `http://127.0.0.1:8000/polls/`

### 2. Entrenar y evaluar el modelo de ML

En otra terminal:

```bash
cd proyecto_testing_ml
python evaluacion_modelo.py
```

Este comando imprime la matriz de confusión, el reporte de clasificación y guarda los archivos `modelo_decision_tree.pkl` y `vectorizer.pkl`.

### 3. Ejecutar la suite completa de pruebas

```bash
pytest -v
```

Ejecuta los 18 tests del prototipo: 8 unitarios del modelo, 7 de integración ML + Selenium y 3 de interfaz web.

## Resultados

### Dataset

- 90 muestras etiquetadas
- 7 categorías de clasificación
- Idioma: inglés (descripciones de comportamiento esperado)

### Categorías

|     Etiqueta   |          Descripción           |
|----------------|--------------------------------|
| list_questions | Visualización de encuestas disponibles |
| question_detail | Acceso al detalle de una encuesta |
| vote_poll      |    Proceso de votación |
| view_results     |   Visualización de resultados |
| missing_choice_error | Manejo de error sin opción seleccionada |
| unpublished_question_access | Acceso a encuestas no publicadas |
| invalid_question_404 | Acceso a encuestas inexistentes |

### Métricas del modelo

- **Accuracy global:** 39%
- **Train/Test split:** 80/20
- **Algoritmo:** Decision Tree (max_depth=10)

### Suite de pruebas

|  Archivo | Tests | Resultado |
|---------|---------|-----------|
| test_model.py | 8 | 8/8 PASSED |
| test_inteligente.py | 7 | 7/7 PASSED |
| test_selenium.py | 3 | 3/3 PASSED |
| **Total** | **18** | **18/18 PASSED** |

## Licencia

Proyecto académico desarrollado en el marco del Programa de Ingeniería de Sistemas de la Universidad de Investigación y Desarrollo (UDI).
