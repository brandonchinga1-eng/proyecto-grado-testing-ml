import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "modelo_decision_tree.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predecir(texto):
    texto_vec = vectorizer.transform([texto])
    prediccion = model.predict(texto_vec)
    return prediccion[0]