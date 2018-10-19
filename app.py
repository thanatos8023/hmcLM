# -*- coding:utf-8 -*-

from flask import Flask
from flask import request, Response
from train import train

app = Flask(__name__)


def decode(sent):
    print("User utterance:", sent)
    intention = train(sent)
    print("Output:", intention)
    return intention


@app.route('/decode', methods=['GET', 'POST'])
def get_intention():
    intend = decode(request.args.get('sent'))
    return Response(intend)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
