from flask import Flask
app = Flask(__name__)
app.debug = True


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_resources(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run()
