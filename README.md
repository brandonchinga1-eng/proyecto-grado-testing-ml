# Proyecto de Grado — Testing de Software con Machine Learning

Prototipo funcional que integra técnicas de Machine Learning (Árbol de Decisión + TF-IDF) con automatización de pruebas (Selenium + Pytest) sobre una aplicación web Django.

## Autor
Brandon Steven Ibáñez Archila — Universidad de Investigación y Desarrollo (UDI)

## Estructura

- django-pools-master/ — Aplicación web Django usada como sistema bajo prueba
- proyecto_testing_ml/ — Módulos de ML, predictor y pruebas automatizadas
  - evaluacion_modelo.py — Entrenamiento y evaluación del clasificador
  - predictor.py — Carga del modelo serializado
  - 	est_model.py — Pruebas unitarias del clasificador (8 tests)
  - 	est_inteligente.py — Pruebas de integración ML + Selenium (7 tests)
  - 	est_selenium.py — Pruebas de interfaz web (3 tests)

## Ejecución

1. Levantar el servidor Django:
   \\\ash
   cd django-pools-master/django-pools-master/django-poll
   python manage.py runserver
   \\\

2. En otra terminal, ejecutar las pruebas:
   \\\ash
   cd proyecto_testing_ml
   python evaluacion_modelo.py
   pytest -v
   \\\

## Resultados

- Dataset: 90 muestras, 7 categorías
- Accuracy del modelo: 39%
- Suite de pruebas: 18/18 PASSED
