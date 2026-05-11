from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# =========================
# DATASET (90 muestras)
# =========================

textos = [
"The homepage shows the latest poll questions",
"The system displays a list of available polls",
"The application lists all published poll questions",
"The user can view the list of current polls",
"The homepage displays the latest questions",
"The system shows recent polls on the main page",
"The user sees a list of poll questions",
"The application displays available polls to vote",
"The platform loads the list of polls on the homepage",
"The web page shows all available poll questions",
"The system retrieves and displays poll questions",
"The main page presents a list of active polls",

"The user can open a poll to see its details",
"The system displays the details of a selected question",
"The application shows the poll question and options",
"The user views the details of a specific poll",
"The system loads the question detail page",
"The poll page shows the available choices",
"The user accesses the detail page of a poll",
"The application displays poll options",
"The system presents the question description and answers",
"The detail page loads the poll with its options",
"The application opens a poll showing its possible answers",
"The user reads the poll question and its choices",

"The user selects an option and submits a vote",
"The system records the vote of a user",
"The user votes for a poll option",
"The application processes a vote submission",
"The user chooses an option and votes",
"The vote is submitted successfully",
"The system registers the selected poll choice",
"The user submits a poll vote",
"The application stores the vote selected by the user",
"The system counts the vote for the selected option",
"The user confirms their vote in the poll",
"The vote request is sent to the server",

"The system shows the results of a poll",
"The user can see how many votes each option received",
"The application displays poll results",
"The results page shows vote counts",
"The user checks the results of a poll",
"The system displays voting results",
"The application shows poll statistics",
"The results page lists votes per option",
"The page displays the total number of votes for each choice",
"The system presents the poll outcome to the user",
"The user reviews the poll result summary",
"The application shows how users voted in the poll",

"The system shows an error when no option is selected",
"The user tries to vote without choosing an option",
"The application prevents submitting an empty vote",
"The system returns an error if no choice is selected",
"The user submits a vote without selecting an option",
"The system displays a validation error for missing choice",
"The vote submission fails if no option is selected",
"The system asks the user to select a choice before voting",
"The application rejects a vote with no selected answer",
"The system warns the user that a choice must be selected",
"The form validation prevents empty vote submission",
"The vote request fails due to missing selection",

"The system prevents access to unpublished polls",
"The user cannot view polls scheduled for the future",
"The application hides unpublished questions",
"The system restricts access to polls not yet published",
"The user attempts to access a poll with a future publication date",
"The system blocks access to unpublished poll questions",
"The application ensures future polls are not visible",
"The user cannot access unpublished content",
"The platform denies access to polls not released yet",
"The system hides polls with a future publication date",
"The user receives an error when accessing unpublished polls",
"The application restricts visibility of future poll questions",

"The system returns a 404 error for an invalid poll ID",
"The user tries to access a poll that does not exist",
"The application displays a 404 page for invalid polls",
"The system shows an error for a non existing poll",
"The user opens a poll URL that does not exist",
"The application returns 404 when the poll is missing",
"The system cannot find the requested poll",
"The application handles invalid poll requests",
"The system reports that the requested poll ID is invalid",
"The user navigates to a poll that is not present in the database",
"The application shows not found when accessing an invalid poll",
"The system displays a page not found error for wrong poll id",
"The user receives a 404 error for a missing poll",
"The application indicates the poll does not exist",
"The server responds with 404 for unknown poll resource",
"The requested poll cannot be located in the system",
"The poll endpoint returns not found for invalid identifier",
"The system reports that the poll resource was not found"
]

etiquetas = (
    ["list_questions"] * 12 +
    ["question_detail"] * 12 +
    ["vote_poll"] * 12 +
    ["view_results"] * 12 +
    ["missing_choice_error"] * 12 +
    ["unpublished_question_access"] * 12 +
    ["invalid_question_404"] * 18
)

# =========================
# PROCESAMIENTO
# =========================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

X_train, X_test, y_train, y_test = train_test_split(
    X, etiquetas, test_size=0.2, random_state=42
)

# max_depth=10 para evitar overfitting
modelo = DecisionTreeClassifier(max_depth=10, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# =========================
# RESULTADOS
# =========================

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_test, y_pred))

# =========================
# GUARDAR MODELO Y VECTORIZADOR
# =========================

joblib.dump(modelo, "modelo_decision_tree.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Modelo y vectorizador guardados correctamente.")
print("   → modelo_decision_tree.pkl")
print("   → vectorizer.pkl")