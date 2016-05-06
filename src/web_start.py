from flask import Flask, request, jsonify
import cPickle as pickle
import math
import itertools
from char_seq_to_latex import char_seq_to_latex
import os
import sys
sys.path.append(os.getcwd())
from src.backend.classifier.convnet import load_model
from src.backend.data_processing.traces2image import traces2image, IMAGE_SIZE


DIST_THRES = 60


app = Flask(__name__)
app.debug = True


def reformat_trace(trace):
    return map(lambda t: (t['x'], t['y']), trace)


def euclidean_dist(ps):
    p1, p2 = ps[0], ps[1]
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def intersect(trace1, trace2):
    shortest_dist = min(map(euclidean_dist, itertools.product(trace1, trace2)))
    return True if shortest_dist < DIST_THRES else False


def segment_traces(traces):
    last_trace = traces[-1]
    ts = [last_trace]
    for trace in reversed(traces[:-1]):
        if not intersect(last_trace, trace):
            break
        ts.append(trace)
        last_trace = trace
    return ts


with open('data/prepared_data/CROHME.pkl', 'rb') as f:
    CROHME = pickle.load(f)
num2sym = CROHME['num2sym']
classifier = load_model("models/convnet/convnet.ckpt")

sym_seq = []

@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_resources(path):
    return app.send_static_file(path)


@app.route('/submit', methods=['POST'])
def submit():
    global sym_seq
    traces = request.json['data']
    traces = map(reformat_trace, traces)
    print traces
    if not traces:
        sym_seq = []
    else:
        ts = segment_traces(traces)
        if len(ts) > 1:
            sym_seq.pop()
        image = traces2image(ts)
        label = num2sym[classifier.predict(image)[0]]

        x_list, y_list = zip(*itertools.chain.from_iterable(traces))
        x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
        sym_seq.append({"char": label, "pos": {"upper": y_min, "lower": y_max, "left": x_min, "right": x_max}})

    latex = char_seq_to_latex(sym_seq)
    return jsonify({'latex': latex})

if __name__ == '__main__':
    app.run()
