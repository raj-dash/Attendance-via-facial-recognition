from flask import Flask, render_template, request, url_for
from facedetectionfinal import *


capture_count_file = "capture_count.txt"
latest_captured_filename = None
capture_count = load_capture_count()
app = Flask(__name__)


# This runs the home page
@app.route('/')
def hello():
    return render_template("index.html", home=True) # Renders the homepage and returns for the browser to load

# When the form is submitted using one of the two buttons on the website, this function is triggered
@app.route('/face_check', methods=['post'])
def face_check():
    result = '' 

    if 'recognise' in request.form:
        if recognize_faces():
            result = "Recognised"
        else:
            result = "Not recognised"
        
    elif 'saveFace' in request.form:
        return render_template('register.html')

    return render_template('result.html', result=result) # Renders the result page and shows whether the face is recognised or not

# when the form for registration of face is submitted
@app.route('/face_register', methods=['post'])
def face_register():
    reg = int(request.form.get("reg_num"))
    print(reg)
    capture_count = load_capture_count()
    latest_captured_filename = capture_and_store_face()

    if latest_captured_filename:
        label = reg
        if label:
            images, labels = [], []
            captured_face = cv2.imread(latest_captured_filename, cv2.IMREAD_GRAYSCALE)
            images.append(captured_face)
            labels.append(int(label))

            recognizer.update(np.asarray(images), np.asarray(labels, dtype=np.int32))
            save_recognizer()
            return render_template('index.html') # placeholder till I make the results page
        else:
            print("No registration number entered. The captured image will not be stored.")

    else:
        print('no file name')
        return 'no file'


if __name__ == "__main__":
    app.run(debug=True)