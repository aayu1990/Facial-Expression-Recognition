# Load libraries
import os
import flask
from flask import Flask, render_template,request, redirect, url_for
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np # linear algebra
from skimage import io
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt, mpld3
from werkzeug import secure_filename

UPLOAD_FOLDER = '/Users/ankurgupta/Desktop/Final-Project/CAM_project3/static/images'
# instantiate flask 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/home")
def main():
    return render_template('index.html')

  
def allowed_file(filename):
    ALLOWED_EXTENSIONS='png'
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result_html = get_output_html(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return '''
                <html>
                    <body>
                        <img src="{0}" alt="Image" width="460" height="345">
                        {1}
                    </body>
                </html>
            '''.format(
                os.path.join("static/images/", filename),
                result_html
            )
            # return get_output_html(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('upload_file',filename=filename))
        
    return '''<!doctype html>
    <title>Upload new File</title>
    <h1>Upload an image to know the expressions</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
		

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc


def emotion_analysis(emotions):
    print(emotions)
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    fig = plt.figure(1)
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    return mpld3.fig_to_html(fig)

# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()


def get_output_html(file_location):
    print('START')
    print(file_location)
    img = image.load_img(file_location, grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    print(x)
    model=load_model("model_ExpReco.h5")
    custom = model.predict(x)
    print(custom)
    html = emotion_analysis(custom[0])
    return '{0}'.format(html)

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.args
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        print('START')
        file_location = params['file_location']
        return get_output_html(file_location)
        # with graph.as_default():
        #     data['custom'] = str(model)
        #     # data["prediction"] = str(model.predict(x)[0][0])
        #     data["success"] = True

    # return a response in json format 
    #return flask.jsonify(data)    
    
# start the flask app, allow remote connections 
app.run(host='0.0.0.0')
