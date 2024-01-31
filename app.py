from __future__ import division, print_function
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import pickle
from nltk.stem import WordNetLemmatizer
import json
import random
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import nltk
lemmatizer = WordNetLemmatizer()
from flask_mysqldb import MySQL

# Keras

# Flask utils

# Define a flask app
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'capstone'

mysql = MySQL(app)

# Model saved with Keras model.save()
MODEL_PATH = 'models/final_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


class_names = {
    1: 'Berawan, Rekomendasi Aktivitas: Bersepeda, Berkebun. Tetap gunakan sunscreen atau sunblock',
    2: 'Hujan, Rekomendasi Aktivitas: Memasak, Membaca Buku. Jangan lupa menggunakan jas hujan atau payung apabila ingin bepergian ke luar rumah',
    3: 'Cerah, Rekomendasi Aktivitas: Berenang, Bermain Golf. Gunakan sunscreen atau sunblock untuk melindungi kulit dari paparan sinar UV',
    4: 'Sunrise, Rekomendasi Aktivitas: Jogging, Yoga. ',
}


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(250, 250))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    pred_class = preds.argmax(axis=-1)            # Simple argmax
    return class_names[pred_class[0]]             # Convert index


model_chat = load_model('model/models.h5')
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts.pkl', 'rb'))
classes = pickle.load(open('model/labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - membagi kata menjadi array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - membuat bentuk singkat untuk kata (kata dasar)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

#bow = mengubah teks menjadi vektor
def bow(sentence, words, show_details=True):
    # sentence -> tokenisasi dan lematisasi
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model_chat):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model_chat.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model_chat)
    res = getResponse(ints, intents)
    return res

@app.route('/', methods=['GET'])
def default():
    return render_template('landing.html')

@app.route('/index', methods=['GET'])
def index():
    # Main page
    return render_template('about.html')


@app.route('/features', methods=['GET'])
def features():
    return render_template('features.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/chatbot', methods=['GET'])
def chatbot():
    return render_template('chatbot.html')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        return result
    return None

# Load SVM model
with open('model/model_svm.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load TfidfVectorizer
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# def preprocess_text(text):
#     # Preprocess the text (e.g., remove special characters, lowercase)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = text.lower()
#     return text

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# @app.route('/sentimen', methods=['POST'])
# def sentimen():
#     if request.method == 'POST':
#         feedback_text = request.form['feedback']
#         # Preprocess the input text
#         preprocessed_text = preprocess_text(feedback_text)
#         # Transform the input text using the TfidfVectorizer
#         input_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray()
#         # Make prediction using the SVM model
#         prediction = svm_model.predict(input_vector)[0]
#         print("Prediction:", prediction)
        
#         sentiment_mapping = {0: 'Negatif', 1: 'Positif'}
#         sentiment = sentiment_mapping[prediction]
 
#         return render_template('feedback.html', feedback=feedback_text, sentiment=sentiment)

@app.route('/sentimen', methods=['POST'])
def sentimen():
    if request.method == 'POST':
        feedback = request.form['feedback']

        # Lakukan analisis sentimen
        sentiment = predict_sentiment(feedback)

        # Simpan ke database
        save_to_database(feedback, sentiment)

         # Hitung persentase hasil sentimen
        positive_percentage, negative_percentage = calculate_sentiment_percentage()
        
        # Ambil data dari database untuk ditampilkan di HTML
        cur = mysql.connection.cursor()
        cur.execute("SELECT feedback_user, hasil_sentimen_analisis FROM feedback ORDER BY id DESC LIMIT 1")
        result = cur.fetchone()
        cur.close()

        user_input = result[0]
        sentiment_result = result[1]

        return render_template('feedback.html', user_input=user_input, sentiment_result=sentiment_result, positive_percentage=positive_percentage, negative_percentage=negative_percentage)
    return render_template('feedback.html')

def predict_sentiment(feedback):
    # Lakukan pra-pemrosesan pada feedback (sesuai dengan langkah preprocessing)
    transformed_feedback = tfidf_vectorizer.transform([feedback])

    # Lakukan prediksi sentimen menggunakan model
    prediction = svm_model.predict(transformed_feedback)

    if prediction[0] in [0]:
        return 'negatif'
    elif prediction[0] in [1]:
        return 'positif'

def save_to_database(feedback, sentiment):
    # Simpan data ke database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO feedback (feedback_user, hasil_sentimen_analisis) VALUES (%s, %s)", (feedback, sentiment))
    mysql.connection.commit()
    cur.close()

def calculate_sentiment_percentage():
    # Ambil jumlah feedback untuk masing-masing sentimen
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='positif'")
    positive_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='negatif'")
    negative_count = cur.fetchone()[0]

    total_count = positive_count + negative_count

    # Hitung persentase
    positive_percentage = (positive_count / total_count) * 100

    negative_percentage = (negative_count / total_count) * 100

    return positive_percentage, negative_percentage

if __name__ == '__main__':
    app.run(debug=True)
