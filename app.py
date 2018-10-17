# -*- coding:utf-8 -*-

from flask import Flask
from flask import request
from train import train

app = Flask(__name__)

@app.route('/test')
def hello_world():
    return "Hello, World!"

@app.route('/decode')
def get_intention():
    sentence = request.args.get('sent')
    print(sentence)
    intention = train(sentence)
    return intention


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
