# -*- coding:utf-8 -*-

from flask import Flask
from flask import request
from decode import decode

app = Flask(__name__)

@app.route('/test')
def hello_world():
    return "Hello, World!"

@app.route('/decode')
def get_intention():
    sentence = request.args.get('sent')
    print(sentence)
    intention = decode(sentence, 'model/hmc.model')
    return intention[0]


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
