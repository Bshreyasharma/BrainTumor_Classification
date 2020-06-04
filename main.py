from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
import model_load_prediction
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
database={'user':'123'}

app = Flask(__name__,static_url_path='/static')

def predict():
    result = model_load_prediction.predict()
    return json.dumps(result)


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/Upload', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            t1 = request.files['t1']
            t1ce = request.files['t1ce']
            flair = request.files['flair']
            t2 = request.files['t2']

            UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
            target = os.path.join(UPLOAD_FOLDER,'dynamic\\')
            upload = os.path.join(target,'upload1\\')

            if not os.path.isdir(target):
                os.mkdir(target)
                os.mkdir(upload)
            t1_filename = secure_filename(t1.filename)
            t1ce_filename = secure_filename(t1ce.filename)
            flair_filename = secure_filename(flair.filename)
            t2_filename = secure_filename(t2.filename)

            UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
            target = os.path.join(UPLOAD_FOLDER,'dynamic\\')
            upload = os.path.join(target,'upload1\\')

            t1.save(os.path.join(upload,t1_filename))
            t1ce.save(os.path.join(upload,t1ce_filename))
            flair.save(os.path.join(upload,flair_filename))
            t2.save(os.path.join(upload,t2_filename))

            json_dict = {"file_path":upload,
                         "flair":flair_filename,
                         "t1":t1_filename,
                         "t2":t2_filename,
                         "t1ce":t1ce_filename}
            with open(upload+"/file_info.json", "w") as outfile:
                json.dump(json_dict, outfile)
        return render_template("BrainTumorClassification.html",load ="yes",result = predict())
    except:
        print('came here')
        return render_template("BrainTumorClassification.html", load=" ",result = "Submit correct type of file")

@app.route('/login',methods=['POST','GET'])
def login():
    return render_template("login.html",info =" ")


@app.route('/loginInput',methods=['POST','GET'])
def loginInput():
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
        return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
            return render_template('BrainTumorClassification.html',load="no", result = " ")

if __name__ == "__main__":
    app.run(debug=True)

