# Importing all of the necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import dash_auth
from users import Username_Password_Pairs

# Instantiate the app
app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, Username_Password_Pairs)
server = app.server

# Importing the data and classification model
data = pd.read_csv("framingham_filled.csv")
model = pickle.load(open("heart_disease_model.pk1", "rb"))

# Dashboard layout
app.layout = html.Div([

    html.H1("Heart Disease Prediction", style={"text-align": "center"}),

    # Dropdown selection for the graph options

    dcc.Dropdown(id="select_graph",
                 options=[
                     {"label": "Blood Pressure Scatterplot", "value": 0},
                     {"label": "Male vs Female", "value": 1},
                     {"label": "Smokers vs Non-smokers", "value": 2},
                     {"label": "Smokers: Cigarettes and Cholesterol", "value": 3},
                     {"label": "Heart Disease by Age", "value": 4}],
                 multi=False,
                 value=0,
                 style={"width": "40%"}
                 ),

    # Graph area
    html.Br(),
    dcc.Graph(id="graph", figure={}),

    html.H1("Heart Disease Predictor"),

    # Input buttons and fields for the classification model
    html.Br(),
    html.Label("Sex: "),
    dcc.RadioItems(id='sex', options=[
        {'label': 'Male', 'value': 1},
        {'label': 'Female', 'value': 0}
    ], value=1),

    html.Br(),
    html.Label("Age: "),
    html.Br(),
    dcc.Input(id="age", type="number", step=1, value=50),
    html.Br(),

    html.Br(),
    html.Label("Smoker status: "),
    dcc.RadioItems(id='smoker', options=[
        {'label': 'Smoker', 'value': 1},
        {'label': 'Non-smoker', 'value': 0}
    ], value=0),

    html.Br(),
    html.Label("Cigarettes per day: "),
    html.Br(),
    dcc.Input(id="cigs", type="number", step=1, value=0),
    html.Br(),

    html.Br(),
    html.Label("On blood pressure meds: "),
    dcc.RadioItems(id='bpmeds', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], value=0),

    html.Br(),
    html.Label("Has had stroke: "),
    dcc.RadioItems(id='stroke', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], value=0),

    html.Br(),
    html.Label("Has hypertension: "),
    dcc.RadioItems(id='hypertension', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], value=0),

    html.Br(),
    html.Label("Has diabetes: "),
    dcc.RadioItems(id='diabetes', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], value=0),

    html.Br(),
    html.Label("Cholesterol: "), html.Br(),
    dcc.Input(id="cholesterol", type="number", step=1, value=235),
    html.Br(),

    html.Br(),
    html.Label("Systolic blood pressure: "), html.Br(),
    dcc.Input(id="systolic", type="number", step=1, value=133),
    html.Br(),

    html.Br(),
    html.Label("Diastolic blood pressure: "), html.Br(),
    dcc.Input(id="diastolic", type="number", step=1, value=83),
    html.Br(),

    html.Br(),
    html.Label("Body Mass Index:"), html.Br(),
    dcc.Input(id="bmi", type="number", step=1, value=26),
    html.Br(),

    html.Br(),
    html.Label("Heart Rate (BPM): "), html.Br(),
    dcc.Input(id="heartrate", type="number", step=1, value=75),
    html.Br(),

    html.Br(),
    html.Label("Glucose: "), html.Br(),
    dcc.Input(id="glucose", type="number", step=1, value=82),
    html.Br(),

    html.Br(),
    html.Label("Heart Disease: "), html.Br(),
    dcc.Textarea(
        id='heartdisease',
        placeholder='Prediction will be here.',
        value='Prediction will be here.',
        style={'width': '100%'},
        disabled=True
    )

])

# Callback for graph function
@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="select_graph", component_property="value")
)

# Method for changing the graph display
def update_graph(option_selected):

    fig = go.Figure()
    dff = data.copy()

    if option_selected == 0:
        hd_no_meds = dff[(dff.HeartDisease == 1) & (dff.BloodPressureMeds == 0)]
        hd_meds = dff[(dff.HeartDisease == 1) & (dff.BloodPressureMeds == 1)]
        nhd_no_meds = dff[(dff.HeartDisease == 0) & (dff.BloodPressureMeds == 0)]
        nhd_meds = dff[(dff.HeartDisease == 0) & (dff.BloodPressureMeds == 1)]

        dff["HeartDisease"] = dff["HeartDisease"].astype(str)

        fig.add_trace(go.Scatter(mode='markers', x=hd_no_meds["SystolicPressure"], y=hd_no_meds["DiastolicPressure"],
                                 opacity=1, marker=dict(color='Red', size=15,
                                 line=dict(color="DarkRed", width=2)), name="Heart Disease (no meds)", showlegend=True))
        fig.add_trace(go.Scatter(mode='markers', x=hd_meds["SystolicPressure"], y=hd_meds["DiastolicPressure"],
                                 opacity=1, marker=dict(color='Red', size=15, symbol="diamond",
                                 line=dict(color="DarkRed", width=2)), name="Heart Disease (meds)", showlegend=True))
        fig.add_trace(go.Scatter(mode='markers', x=nhd_no_meds["SystolicPressure"], y=nhd_no_meds["DiastolicPressure"],
                                 opacity=0.3, marker=dict(color='Blue', size=15,
                                 line=dict(color="DarkBlue", width=2)), name="No Heart Disease (no meds)", showlegend=True))
        fig.add_trace(go.Scatter(mode='markers', x=nhd_meds["SystolicPressure"], y=nhd_meds["DiastolicPressure"],
                                 opacity=0.3, marker=dict(color='Blue', size=15, symbol="diamond",
                                 line=dict(color="DarkBlue", width=2)), name="No Heart Disease (meds)", showlegend=True))

        fig.update_layout(height=600, width=1200, title="Blood Pressure (X = Systolic, Y = Diastolic)")

    elif option_selected == 1:
        m_dia_hd = dff[(dff.Male == 1) & (dff.Diabetes == 1) & (dff.HeartDisease == 1)]
        m_dia_nhd = dff[(dff.Male == 1) & (dff.Diabetes == 1) & (dff.HeartDisease == 0)]
        m_ndia_hd = dff[(dff.Male == 1) & (dff.Diabetes == 0) & (dff.HeartDisease == 1)]
        m_ndia_nhd = dff[(dff.Male == 1) & (dff.Diabetes == 0) & (dff.HeartDisease == 0)]
        f_dia_hd = dff[(dff.Male == 0) & (dff.Diabetes == 1) & (dff.HeartDisease == 1)]
        f_dia_nhd = dff[(dff.Male == 0) & (dff.Diabetes == 1) & (dff.HeartDisease == 0)]
        f_ndia_hd = dff[(dff.Male == 0) & (dff.Diabetes == 0) & (dff.HeartDisease == 1)]
        f_ndia_nhd = dff[(dff.Male == 0) & (dff.Diabetes == 0) & (dff.HeartDisease == 0)]

        last_count = f_ndia_nhd.shape[0]

        labels = ["Diabetes + Heart Disease", "Diabetes Only", "Heart Disease Only", "Neither"]
        m_values = [m_dia_hd["Male"].sum(), m_dia_nhd["Male"].sum(), m_ndia_hd["Male"].sum(), m_ndia_nhd["Male"].sum()]
        f_values = [f_dia_hd["HeartDisease"].sum(), f_dia_nhd["Diabetes"].sum(), f_ndia_hd["HeartDisease"].sum(), last_count]

        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=m_values, name="Male"), 1, 1)
        fig.add_trace(go.Pie(labels=labels, values=f_values, name="Female"), 1, 2)
        fig.update_traces(hole=0.4)
        fig.update_layout(title_text="Male vs. Female", height=600, width=1200,
                          annotations=[dict(text='Male', x=0.2, y=0.5, font_size=20, showarrow=False),
                                       dict(text='Female', x=0.8, y=0.5, font_size=20, showarrow=False)])

    elif option_selected == 2:
        smoker = dff[dff["Smoker"] == 1]
        nonsmoker = dff[dff["Smoker"] == 0]
        fig.add_trace(go.Histogram(x=nonsmoker["HeartDisease"], name="Non-Smoker"))
        fig.add_trace(go.Histogram(x=smoker["HeartDisease"], name="Smokers"))
        fig.update_traces(opacity=0.6)

        fig.update_layout(bargap=0.5, height=600, width=1200, title="Smoker vs Non-smoker (X = Heart Disease, Y = People)")

    elif option_selected == 3:
        hd_smokers = dff[(dff.Smoker == 1) & (dff.HeartDisease == 1)]
        nhd_smokers = dff[(dff.Smoker == 1) & (dff.HeartDisease == 0)]

        fig.add_trace(go.Scatter(mode='markers', x=hd_smokers["CigarettesPerDay"], y=hd_smokers["TotalCholesterol"],
                                 opacity=1, marker=dict(color='Red', size=15,
                                 line=dict(color='DarkRed', width=2)), name="Heart Disease", showlegend=True))
        fig.add_trace(go.Scatter(mode='markers', x=nhd_smokers["CigarettesPerDay"], y=nhd_smokers["TotalCholesterol"],
                                 opacity=0.3, marker=dict(color='Blue', size=15,
                                 line=dict(color='DarkBlue', width=2)), name="No Heart Disease", showlegend=True))

        fig.update_layout(height=600, width=1200, title="Cigarettes and Cholesterol (X = Cigarettes, Y = Cholesterol)")

    elif option_selected == 4:
        hd = dff[(dff.HeartDisease == 1)]

        fig.add_trace(go.Histogram(x=hd["Age"], nbinsx=37))
        fig.update_layout(height=600, width=1200, title="Heart Disease by Age (X = Age, Y = People)", bargap=0.2)

    return fig

# Callback for the prediction model
@app.callback(
    Output(component_id='heartdisease', component_property='value'),
    Input(component_id='sex', component_property='value'),
    Input(component_id='age', component_property='value'),
    Input(component_id='smoker', component_property='value'),
    Input(component_id='cigs', component_property='value'),
    Input(component_id='bpmeds', component_property='value'),
    Input(component_id='stroke', component_property='value'),
    Input(component_id='hypertension', component_property='value'),
    Input(component_id='diabetes', component_property='value'),
    Input(component_id='cholesterol', component_property='value'),
    Input(component_id='systolic', component_property='value'),
    Input(component_id='diastolic', component_property='value'),
    Input(component_id='bmi', component_property='value'),
    Input(component_id='heartrate', component_property='value'),
    Input(component_id='glucose', component_property='value')
)

# Method for the prediction
def predict(sex, age, smoker, cigs, bpmeds, stroke, hypertension, diabetes, cholesterol, systolic, diastolic, bmi, heartrate, glucose):

    prediction = ""
    patient = pd.DataFrame({
        "Male": sex,
        "Age": age,
        "Smoker": smoker,
        "CigarettesPerDay": cigs,
        "BloodPressureMeds": bpmeds,
        "Stroke": stroke,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "TotalCholesterol": cholesterol,
        "SystolicPressure": systolic,
        "DiastolicPressure": diastolic,
        "BodyMassIndex": bmi,
        "HeartRate": heartrate,
        "Glucose": glucose,
    }, index=[0])
    pred = model.predict(patient)
    prob = model.predict_proba(patient)
    if pred == 0:
        prediction = "NO - Heart Disease is not likely within the next 10 years.\n" + "Probability (No/Yes): " + str(prob)
    elif pred == 1:
        prediction = "YES - This person is at risk for heart disease within the next 10 years.\n" + "Probability (No/Yes): " + str(prob)
    return prediction

if __name__ == "__main__":
    app.run_server(debug=True)