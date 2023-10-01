from flask import Flask, request, render_template
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from dotenv import load_dotenv
import os
import string

load_dotenv()

from flask_sqlalchemy import SQLAlchemy

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

clf = load('../model.joblib')
vectorizer = load('../vectorizer.joblib')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    # Stem or lemmatize the words
    words = [stemmer.stem(word) for word in words]
   
        # Join the words back into a string
    text = ' '.join(words)

    return text


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')

db = SQLAlchemy(app)

class Article(db.Model):
    __tablename__ = 'articles'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    classification = db.Column(db.Text)

    def __init__(self, text, classification):
        self.text = text
        self.classification = classification


@app.route('/')
def home():
    articles = Article.query.order_by(Article.id.desc()).limit(5).all()

    return render_template('home.html', articles= articles)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_text = preprocess_text(text)
    X = vectorizer.transform([preprocessed_text])
    y_pred = clf.predict(X)
    if y_pred[0]== 1:
        result = 'real'
    else:
        result = 'fake'

    article = Article(text, result)
    db.session.add(article)
    db.session.commit()

    return render_template('result.html', result=result, text=text)

if __name__ == '__main__':
    app.run(debug=True)