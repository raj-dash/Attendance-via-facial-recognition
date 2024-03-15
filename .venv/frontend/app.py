from flask import Flask, render_template, request, url_for

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html", home=True)

@app.route('/face_check', methods=['post'])
def face_check():
    opt = ''

    if 'recognise' in request.form:
        opt = 'recognise'
    elif 'saveFace' in request.form:
        opt = 'saveFace'

    return render_template('result.html', result=opt)


if __name__ == "__main__":
    app.run(debug=True)