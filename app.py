from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

preprocessor = bundle['preprocessor']
model = bundle['model']
columns = bundle['columns'] 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Age': float(request.form['Age']),
        'Weight': float(request.form['Weight']),
        'Height': float(request.form['Height']),
        'BMI': float(request.form['BMI']),
        'Menstrual_Regularity': request.form['Menstrual_Regularity'],
        'Menstrual_Cycle_Length': request.form['Menstrual_Cycle_Length'],
        'Irregular_Periods': request.form['Irregular_Periods'],
        'Family_History_PCOS': request.form['Family_History_PCOS'],
        'Excessive_Hair_Growth': request.form['Excessive_Hair_Growth'],
        'Acne_Severity': request.form['Acne_Severity'],
        'Hair_Loss': request.form['Hair_Loss'],
        'Dark_Patches': request.form['Dark_Patches'],
        'Fatigue': request.form['Fatigue'],
        'Mood_Swings': request.form['Mood_Swings'],
        'Weight_Gain': request.form['Weight_Gain'],
        'Fast_Food_Frequency': request.form['Fast_Food_Frequency'],
        'Exercise_Days': request.form['Exercise_Days'],
        'Stress_Level': request.form['Stress_Level']
    }

    input_df = pd.DataFrame([data])

    input_encoded = preprocessor.transform(input_df)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0, 1] * 100

    result = f"Yes, {probability:.2f}% risk" if prediction == 1 else f"No, {probability:.2f}% risk"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
