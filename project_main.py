from logging import debug
from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np

model = pickle.load(open('model/model.pkl','rb'))

app = Flask(__name__)

###########################################################################################
############################ Creating API App ##############################################
###########################################################################################

@app.route("/")
def species_prediction():

    return render_template("index.html")

################################################################################################

@app.route("/iris_project",methods = ["POST"])
def project_predict():
    SepalLengthCm = float(request.form.get('SepalLengthCm'))
    SepalWidthCm = float(request.form.get('SepalWidthCm'))
    PetalLengthCm = float(request.form.get('PetalLengthCm'))
    PetalWidthCm = float(request.form.get('PetalWidthCm'))


    result = model.predict(np.array([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]).reshape(1,4))
        

    return render_template('index.html',prediction = result[0])

#####################################################################################################


if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 8080, debug=True)