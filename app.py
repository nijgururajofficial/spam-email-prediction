from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('savedmodels/model.pkl','rb'))
cv = pickle.load(open('savedmodels/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result',methods=['GET', 'POST'])
def result():
    feature = request.form.values()
    final = cv.transform(feature)
    result = model.predict(final)
    return render_template('result.html',result = result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)