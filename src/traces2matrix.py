MATRIX_WIDTH = 20
MATRIX_HEIGHT = 20
MATRIX_PADDING = 1


def traces2matrix(traces):
    matrix = [[0 for x in range(MATRIX_WIDTH)] for x in range(MATRIX_HEIGHT)]
    # Compute original scale
    max_x = traces[0][0][0]
    min_x = traces[0][0][0]
    max_y = traces[0][0][1]
    min_y = traces[0][0][1]
    for trace in traces:
        for point in trace:
            x = point[0]
            max_x = max(x, max_x)
            min_x = min(x, min_x)
            y = point[1]
            max_y = max(y, max_y)
            min_y = min(y, min_y)

    # Fill in scaled matrix
    original_scale_x = max_x - min_x
    original_scale_y = max_y - min_y
    for trace in traces:
        for point in trace:
            x = point[0]
            portion_x = (x - min_x) / original_scale_x
            scaled_x = (MATRIX_WIDTH - MATRIX_PADDING * 2) * portion_x + MATRIX_PADDING
            y = point[1]
            portion_y = (y - min_y) / original_scale_y
            scaled_y = (MATRIX_HEIGHT - MATRIX_PADDING * 2) * portion_y + MATRIX_PADDING
            matrix[scaled_y][scaled_x] = 1

    return matrix

if __name__ == '__main__':
    test_traces = [[(351, 57), (351, 57), (351, 56), (350, 56)], [(383, 75), (383, 75), (383, 73)]]
    print traces2matrix(test_traces)
