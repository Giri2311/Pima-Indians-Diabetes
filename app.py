#import relavent libraries for flask ,html rendering and loading the ML Model
from flask import Flask,request,url_for,redirect,render_template
import pickle
import joblib
from matplotlib import transforms
import pandas as pd
app= Flask(__name__)

# model= pickle.load(open("model.pkl","rb"))
# scale= pickle.load(open("scale.pkl","rb"))

model= joblib.load(open("model.pkl","rb"))
scale= joblib.load(open("scale.pkl","rb"))


@app.route("/")
def landingPage():
    return render_template('index.html')
@app.route("/predict",methods=["POST"])
def predict():
    pregnancies= request.form['1']
    glucose= request.form['2']
    bloodpressure= request.form['3']
    skinthickness= request.form['4']
    insulin= request.form['5']
    bmi= request.form['6']
    dpf= request.form['7']
    age= request.form['8']

    rowDF= pd.DataFrame([pd.Series([pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,dpf,age])])
    rowDF_new= pd.DataFrame(scale.transform(rowDF))
    print(rowDF_new)

    #model prediction 
    prediction= model.predict_proba(rowDF_new)
    

    if prediction[0][1] >0.5:
        valPred= round(prediction[0][1],3)
        return render_template('result.html',pred= f"You have a chance of having diabetes.\n\nProbability of you being diabetic is: {prediction[0][1]*100:.2f}%")
    else:
        valPred= round(prediction[0][0],3)
        return render_template('result.html',pred= f"You appear to be safe from Diabetes.\n\n Probability of you being non-diabetic is: {prediction[0][0]*100:.2f}%")


if __name__ == "__main__":
    app.run(debug=True)

