from flask import Flask, request, jsonify
app = Flask(__name__)
app.debug = True
from char_seq_to_latex import char_seq_to_latex


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_resources(path):
    return app.send_static_file(path)


@app.route('/submit', methods=['POST'])
def submit():
    traces = request.json['data']
    print traces
    # char_seq = [{'char': 'x', 'pos': (0, 0)}, {'char': '=', 'pos': (0, 0)}, {'char': '5', 'pos': (0, 0)}]
    # latex = char_seq_to_latex(char_seq)
    return jsonify({'latex': 'E=mc^2'})

if __name__ == '__main__':
    app.run()
