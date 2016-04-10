import xml.etree.ElementTree as ET
import glob
import numpy as np
from traces2image import IMAGE_WIDTH, IMAGE_HEIGHT, traces2image

DATA_SOURCE = "data/TC11_package/CROHME2014_data/isolatedTest2014/"
GT_FILE = "iso_GT.txt"
EXT = ".inkml"


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
        stroke = map(lambda x: map(int, x.strip().split(' ')), trace.text.split(','))
        trace_list.append(stroke)
    return ink_id, trace_list


def load_int_data(inkml_dir, ink_id_map):
    X = np.empty((len(ink_id_map), IMAGE_HEIGHT * IMAGE_WIDTH))
    for inkml_file in glob.glob(DATA_SOURCE + "*.inkml"):
        ink_id, trace_list = load_symbol(inkml_file)
        r = ink_id_map.get(ink_id)
        if r is None:
            continue
        X[r, :] = traces2image(trace_list).reshape((1, -1))
    return X


if __name__ == "__main__":
    y, symbol_set, ink_id_map = load_ground_truth(DATA_SOURCE + GT_FILE)