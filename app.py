# -*- coding:utf-8 -*-

from flask import Flask
from flask import request, Response
from train import train

app = Flask(__name__)


@app.route('/decode')
def get_intention():
    sent = request.args.get('sent')
    print("User utterance:", sent)
    intention = train(sent)
    print("Output:", intention)

    return intention[0]



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
