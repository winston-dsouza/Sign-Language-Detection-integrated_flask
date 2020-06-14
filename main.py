from flask import Flask, render_template, Response,request,redirect,url_for,make_response,jsonify
from motion import VideoCameraMotion
import time

import itertools
too=itertools.cycle([True,False])
def foo():
    return next(too)

def obj(o):
    j=o.predictor()
    return j


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

@app.route('/',methods=['GET', 'POST'])
@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route('/motion')
def motion():
    return render_template('motion.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_motion')
def video_motion():
    return Response(gen(VideoCameraMotion()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/instruction')
def instruction():

    return render_template('instruction.html')

@app.route('/dashboard')
def dashboard():

    return render_template('dashboard.html')

@app.route('/toogle')
def toogle():
    VideoCameraMotion.flag=foo()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
 