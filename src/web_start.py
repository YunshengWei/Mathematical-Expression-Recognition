from flask import Flask, request, jsonify
app = Flask(__name__)
app.debug = True


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_resources(path):
    return app.send_static_file(path)


@app.route('/submit', methods=['POST'])
def submit():
    print request.json['data']
    return jsonify(success=True)

if __name__ == '__main__':
    app.run()
