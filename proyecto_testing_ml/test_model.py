from predictor import predecir

# Pruebas de clasificación por categoría

def test_list_questions():
    resultado = predecir("The homepage shows the latest poll questions")
    assert resultado == "list_questions"

def test_question_detail():
    resultado = predecir("The user views the details of a specific poll")
    assert resultado == "question_detail"

def test_vote_poll():
    resultado = predecir("The user selects an option and submits a vote")
    assert resultado == "vote_poll"

def test_view_results():
    resultado = predecir("The system shows the results of a poll")
    assert resultado == "view_results"

def test_missing_choice_error():
    resultado = predecir("The system shows an error when no option is selected")
    assert resultado == "missing_choice_error"

def test_unpublished_question_access():
    resultado = predecir("The system prevents access to unpublished polls")
    assert resultado == "unpublished_question_access"

def test_invalid_question_404():
    resultado = predecir("The system returns a 404 error for an invalid poll ID")
    assert resultado == "invalid_question_404"

def test_prediccion_no_nula():
    """Smoke test: el modelo responde sin errores ante cualquier entrada"""
    resultado = predecir("some random text input")
    assert resultado is not None