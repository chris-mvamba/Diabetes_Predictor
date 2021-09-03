import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

app.config['SECRET_KEY'] = '9a3e10786452a6d9b00f0b3f86b285b4'

model = pickle.load(open('model_1.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 4, 5, 7]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_data = [float(x) for x in request.form.values()]
    
    new_input = np.array(input_data)
    new_input = new_input.reshape(1,-1)
    prediction = model.predict( sc.transform(new_input) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
