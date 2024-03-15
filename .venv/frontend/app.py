from flask import Flask, render_template, request, url_for


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
        result = 'recognise'
    elif 'saveFace' in request.form:
        result = 'saveFace'

    return render_template('result.html', result=result) # Renders the result page and shows whether the face is recognised or not


if __name__ == "__main__":
    app.run(debug=True)