import joblib
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Load models and scaler
model_lr = joblib.load('models/model_lr.pkl')
model_rf = joblib.load('models/model_rf.pkl')
model_svm = joblib.load('models/model_svm.pkl')
scaler = joblib.load('models/scaler.pkl')

# Feature abbreviation to full name mapping
feature_labels = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (Oldpeak)',
    'slope': 'Slope of ST Segment',
    'ca': 'Major Vessels Colored',
    'thal': 'Thalassemia'
}

# Feature order
feature_names = list(feature_labels.keys())

# Input component generator
def get_input_component(feature):
    dropdown_options = {
        'sex': [{"label": "Male", "value": 1}, {"label": "Female", "value": 0}],
        'cp': [{"label": "Typical Angina", "value": 1},
               {"label": "Atypical Angina", "value": 2},
               {"label": "Non-anginal Pain", "value": 3},
               {"label": "Asymptomatic", "value": 0}],
        'fbs': [{"label": "True", "value": 1}, {"label": "False", "value": 0}],
        'restecg': [{"label": "Normal", "value": 1},
                    {"label": "ST-T Abnormality", "value": 2},
                    {"label": "Hypertrophy", "value": 0}],
        'exang': [{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
        'slope': [{"label": "Upsloping", "value": 2},
                  {"label": "Flat", "value": 1},
                  {"label": "Downsloping", "value": 0}],
        'thal': [{"label": "Normal", "value": 2},
                 {"label": "Fixed Defect", "value": 1},
                 {"label": "Reversible Defect", "value": 3}]
    }

    if feature in dropdown_options:
        return dcc.Dropdown(
            id=f"input-{feature}",
            options=dropdown_options[feature],
            className="form-control",
            placeholder="Select"
        )
    else:
        return dcc.Input(
            id=f"input-{feature}",
            type="number",
            step=0.01,
            className="form-control",
            placeholder="Enter value"
        )

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("Heart Disease Prediction", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label(f"{feature.upper()} - {feature_labels[feature]}"),
            get_input_component(feature)
        ], width=6) for feature in feature_names
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col([
            dbc.Label("Select Model"),
            dcc.Dropdown(id="model-dropdown",
                         options=[
                             {"label": "Logistic Regression", "value": "lr"},
                             {"label": "Random Forest", "value": "rf"},
                             {"label": "SVM", "value": "svm"}
                         ],
                         value="lr",
                         className="form-control")
        ], width=6)
    ]),

    dbc.Button("Predict", id="predict-button", color="primary", className="mt-3"),

    html.Div(id="output", className="mt-4")
])

# Callback
@app.callback(
    Output("output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{feature}", "value") for feature in feature_names] +
    [State("model-dropdown", "value")]
)
def predict(n_clicks, *args):
    if not n_clicks:
        return ""

    values = args[:-1]
    selected_model = args[-1]

    if None in values:
        return dbc.Alert("Please fill in all input values.", color="warning")

    try:
        input_data = scaler.transform([values])
        model = {"lr": model_lr, "rf": model_rf, "svm": model_svm}.get(selected_model)

        if not model:
            return dbc.Alert("Invalid model selected.", color="danger")

        prediction = model.predict(input_data)[0]
        result_text = "Heart Disease" if prediction == 1 else "No Heart Disease"
        result_color = "danger" if prediction == 1 else "success"

        return dbc.Alert(f"Prediction: {result_text}", color=result_color)

    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051, host='0.0.0.0')

# Your full Dash code goes here
# Paste your final app.py code here between triple quotes
