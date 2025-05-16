
import joblib
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Load models
model_lr = joblib.load('models/model_lr.pkl')
model_rf = joblib.load('models/model_rf.pkl')
model_svm = joblib.load('models/model_svm.pkl')
scaler = joblib.load('models/scaler.pkl')

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

feature_labels = {
    'age': 'Age (years)',
    'sex': 'Sex (1 = male, 0 = female)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 (1 = true, 0 = false)',
    'restecg': 'Resting ECG (0 = hypertrophy, 1 = normal, 2 = abnormal)',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (1 = yes, 0 = no)',
    'oldpeak': 'ST Depression (relative to rest)',
    'slope': 'Slope of ST Segment (0 = down, 1 = flat, 2 = up)',
    'ca': 'Number of Major Vessels (0–3)',
    'thal': 'Thalassemia (1 = fixed, 2 = normal, 3 = reversible)'
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For Render deployment

app.layout = dbc.Container([
    html.H1("Heart Disease Prediction", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Label(feature_labels[feature]),
            dcc.Input(id=f"input-{i}", type="number", step=0.01, className="form-control")
        ], width=4) for i, feature in enumerate(feature_names)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Model"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[
                    {"label": "Logistic Regression", "value": "lr"},
                    {"label": "Random Forest", "value": "rf"},
                    {"label": "SVM", "value": "svm"}
                ],
                value="lr", className="form-control"
            )
        ], width=4)
    ]),
    dbc.Button("Predict", id="predict-button", color="primary", className="mt-3"),
    html.Div(id="output", className="mt-4")
])

@app.callback(
    Output("output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{i}", "value") for i in range(len(feature_names))] +
    [State("model-dropdown", "value")]
)
def predict(n_clicks, *args):
    if not n_clicks:
        return ""
    values = args[:-1]
    selected_model = args[-1]
    if None in values:
        return dbc.Alert("Please enter all input values.", color="warning")

    input_data = scaler.transform([values])

    if selected_model == "lr":
        model = model_lr
    elif selected_model == "rf":
        model = model_rf
    elif selected_model == "svm":
        model = model_svm
    else:
        return dbc.Alert("Invalid model selected.", color="danger")

    prediction = model.predict(input_data)[0]
    result_text = "Heart Disease" if prediction == 1 else "No Heart Disease"
    result_color = "danger" if prediction == 1 else "success"

    return dbc.Alert(f"Prediction: {result_text}", color=result_color)

if __name__ == '__main__':
    app.run(debug=True)
