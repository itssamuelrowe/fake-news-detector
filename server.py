from flask import Flask, render_template, request
import pickle

app = Flask(__name__, static_folder="./static")

load_model = pickle.load(open('final_model.sav', 'rb'))

def detecting_fake_news(message):
    # Retrieving the best model for prediction call
    prediction = load_model.predict([ message ])
    print(prediction)
    print("The given statement is ", prediction[0])
    return prediction[0]

@app.route('/review', methods=['POST'])
def review():
    text = request.form['text']
    result = detecting_fake_news(text)
    return render_template('index.html', title='Result | Fake News Detector', result=result)

@app.route('/')
def index():
    return render_template('index.html', title='Fake News Detector')

app.run(host='0.0.0.0', port=80)