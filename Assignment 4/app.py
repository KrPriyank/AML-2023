#imports
import pickle
import score
from flask import Flask, request, render_template, url_for, redirect

#setting names
app = Flask(__name__)
fname = open("mlp",'rb')
model = pickle.load(fname)
threshold = 0.5

@app.route('/') 
def home():
    return render_template('spam.html')

@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label, prop = score.score(sent,model,threshold)
    input = "Spam" if label == 1 else "not spam"
    output = f"""The sentence "{sent}" is {input} with propensity {prop}."""
    return render_template('res.html', ans=output)

if __name__ == '__main__': 
    app.run(debug=True)