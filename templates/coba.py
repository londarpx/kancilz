from flask import Flask, render_template, Response
import cv2

app = Flask(__name__,template_folder='templateFiles', static_folder='staticFiles')

camera = cv2.VideoCapture(0)  


def gen_frames():  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)