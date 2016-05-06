import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
import scipy
sys.path.append(os.getcwd())
from src.backend.classifier import convnet

IMAGE_SIZE = 28
nn = convnet.load_model('models/convnet/convnet.ckpt')


def get_image_data(crohme, image_data, image_index):
    ground_truth_label = crohme['num2sym'][np.argmax(image_data[1][image_index])]
    img = image_data[0][image_index].reshape(IMAGE_SIZE, IMAGE_SIZE)

    predicted_label = crohme['num2sym'][nn.predict(img)[0]]
    return img, ground_truth_label, predicted_label


def gather_image_information(crohme, all_data):
    predicted_label_image_dict = {}
    ground_truth_label_image_dict = {}

    for data in all_data:
        for i in range(0, len(data[0])):
            image_index = i
            img, ground_truth_label, predicted_label = get_image_data(crohme, data, image_index)

            if predicted_label not in predicted_label_image_dict:
                predicted_label_image_dict[predicted_label] = []
            predicted_label_image_dict[predicted_label].append((img, ground_truth_label))

            if ground_truth_label not in ground_truth_label_image_dict:
                ground_truth_label_image_dict[ground_truth_label] = []
            ground_truth_label_image_dict[ground_truth_label].append((img, predicted_label))

    return predicted_label_image_dict, ground_truth_label_image_dict


def show_image(img, ground_truth_label, predicted_label):
    figure = plt.figure()

    ax = figure.add_subplot(111)
    figure.subplots_adjust(top=0.85)
    ax.set_title('predicted label: ' + predicted_label)

    imgplot = plt.imshow(img, cmap="gray", label=predicted_label)
    figure.suptitle('ground truth label:' + ground_truth_label, fontsize=16, fontweight='bold')
    plt.imshow(img, cmap="gray")


if __name__ == '__main__':
    with open('data/prepared_data/CROHME.pkl', 'rb') as f:
        CROHME = pickle.load(f)

        train, val, test = CROHME['train'], CROHME['val'], CROHME['test']
        # all_data = [test]
        all_data = [train, val, test]

        predicted_label_image_dict, ground_truth_label_image_dict = gather_image_information(CROHME, all_data)

        plt.ion()
        while True:
            label = str(raw_input("Press input a label.\n"))
            if label in ground_truth_label_image_dict:
                results = ground_truth_label_image_dict[label]
                for result in results:
                    img, predicted_label = result
                    show_image(img, label, predicted_label)
                    plt.show()
                    _ = raw_input("Press keys to continue.")
                    plt.close()
            else:
                pass

        # plt.ion()
        # for i in range(10, 20):
        #     img, ground_truth_label, predicted_label = get_image_data(crohme, image_data, image_index)
        #     show_image(img, ground_truth_label, predicted_label)
        #     plt.show()
        #     _ = raw_input("Press keys to continue.")
        #     plt.close()