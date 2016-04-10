import numpy as np
from itertools import chain
from scipy.misc import imresize

IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30
IMAGE_PADDING = 3


def traces2image(traces):
    x_list, y_list = zip(*chain.from_iterable(traces))
    x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
    height, width = y_max - y_min + 1, x_max - x_min + 1
    image = np.zeros((height, width))

    for trace in traces:
        last_x, last_y = None, None
        for x, y in trace:
            if last_x:
                if last_x == x:
                    yy_list = range(y + 1, last_y) if last_y > y else range(last_y + 1, y)
                    xx_list = [x] * len(yy_list)
                else:
                    slope, bias = (last_y - y) / float(last_x - x), (last_x * y - last_y * x) / float(last_x - x)
                    xx_list = range(x + 1, last_x) if last_x > x else range(last_x + 1, x)
                    yy_list = map(lambda x: int(round(x * slope + bias)), xx_list)
                    if slope != 0:
                        yy_list2 = range(y + 1, last_y) if last_y > y else range(last_y + 1, y)
                        xx_list2 = map(lambda y: int(round((y - bias) / slope)), yy_list2)
                        xx_list.extend(xx_list2)
                        yy_list.extend(yy_list2)

                for xx, yy in zip(xx_list, yy_list):
                    image[yy - y_min, xx - x_min] = 1

            image[y - y_min, x - x_min] = 1
            last_x, last_y = x, y

    if height < width:
        top_padding = (width - height) / 2
        bottom_padding = width - height - top_padding
        image = np.vstack([np.zeros((top_padding, width)), image, np.zeros((bottom_padding, width))])
    elif width < height:
        left_padding = (height - width) / 2
        right_padding = height - width - left_padding
        image = np.hstack([np.zeros((height, left_padding)), image, np.zeros((height, right_padding))])

    image = imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), 'bilinear')
    #image[image > 0] = 255
    image = np.vstack([np.zeros((IMAGE_PADDING, 2 * IMAGE_PADDING + IMAGE_WIDTH)),
        np.hstack([np.zeros((IMAGE_HEIGHT, IMAGE_PADDING)), image, np.zeros((IMAGE_HEIGHT, IMAGE_PADDING))]),
        np.zeros((IMAGE_PADDING, 2 * IMAGE_PADDING + IMAGE_WIDTH))])

    return image


if __name__ == "__main__":
    from prepare_data import load_symbol, DATA_SOURCE, load_ground_truth, GT_FILE
    import matplotlib.pyplot as plt
    y, symbol_set, ink_id_map = load_ground_truth(DATA_SOURCE + GT_FILE)
    for i in xrange(2050, 2150):
        a,b = load_symbol(DATA_SOURCE + 'iso%s.inkml' % i)
        j = ink_id_map.get(a)
        c = traces2image(b)
        if j != None:
            print i, y[j], np.max(c)
        plt.imshow(c, cmap='gray')
        plt.show()
        #plt.close()