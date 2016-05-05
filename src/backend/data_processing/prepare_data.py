import glob
import xml.etree.ElementTree as ET

import numpy as np
import cPickle as pickle
from sklearn.cross_validation import train_test_split

from traces2image import IMAGE_SIZE, traces2image

DATA_SOURCE = "data/TC11_package/CROHME2014_data/isolatedTest2014/"
GT_FILE = "iso_GT.txt"


def load_ground_truth(gt_file):
    y = []
    symbol_set = set()
    ink_id_map = {}

    with open(gt_file, 'r') as f:
        for line in f:
            ink_id, sym = map(lambda x: x.strip(), line.strip().split(','))
            # Ignore invalid symbol for simplicity
            if sym == "junk":
                continue

            y.append(sym)
            symbol_set.add(sym)
            ink_id_map[ink_id] = len(y) - 1
    return y, symbol_set, ink_id_map


def load_symbol(inkml_file):
    ns = "http://www.w3.org/2003/InkML"
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    ink_id = filter(lambda e: e.get('type') == "UI",
                     root.findall("{%s}annotation" % ns))[0].text
    trace_list = []
    for trace in root.iter("{%s}trace" % ns):
        stroke = map(lambda x: map(lambda i: int(round(float(i))), x.strip().split(' ')), trace.text.split(','))
        trace_list.append(stroke)
    return ink_id, trace_list


def load_int_data(inkml_dir, ink_id_map):
    X = np.empty((len(ink_id_map), IMAGE_SIZE * IMAGE_SIZE))
    for inkml_file in glob.glob(DATA_SOURCE + "*.inkml"):
        ink_id, trace_list = load_symbol(inkml_file)
        r = ink_id_map.get(ink_id)
        if r is None:
            continue
        X[r, :] = traces2image(trace_list).reshape((1, -1))
    return X


def build_symbol_map(symbol_set):
    sym2num = {s: i for i, s in enumerate(symbol_set)}
    num2sym = {i: s for i, s in enumerate(symbol_set)}
    return sym2num, num2sym


def train_val_test_split(X, y):
    # train : val : test = 7 : 1.5 : 1.5
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1.5/8.5)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    y, symbol_set, ink_id_map = load_ground_truth(DATA_SOURCE + GT_FILE)
    X = load_int_data(DATA_SOURCE, ink_id_map)

    sym2num, num2sym = build_symbol_map(symbol_set)
    y_ = np.zeros((len(y), len(symbol_set)), dtype=np.int)
    for i, s in enumerate(y):
        y_[i, sym2num[s]] = 1

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y_)
    CROHME = {"sym2num": sym2num,
              "num2sym": num2sym,
              "train": [X_train, y_train],
              "val": [X_val, y_val],
              "test": [X_test, y_test]}
    with open('data/prepared_data/CROHME.pkl', 'wb') as f:
        pickle.dump(CROHME, f, -1)
