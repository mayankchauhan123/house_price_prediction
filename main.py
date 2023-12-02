import pandas as pd
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bath = request.form.get('bath')
    total_sqft = request.form.get('total_sqft')
    bhk = request.form.get('bhk')

    print(location,bath,total_sqft,bhk)
    input = pd.DataFrame([[location,bath,total_sqft,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0]

    return str(prediction)

if __name__== "__main__":
    app.run(debug=True,port=5001)