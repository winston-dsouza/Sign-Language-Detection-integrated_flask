from flask import Flask, render_template, Response,request
from camera import VideoCamera
from motion import VideoCameraMotion

app = Flask(__name__)

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


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_motion')
def video_motion():
    return Response(gen(VideoCameraMotion()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/instruction')
def instruction():

    return render_template('instruction.html')

@app.route('/dashboard')
def dashboard():

    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)