def char_seq_to_latex(char_seq):
    """
    Converts a char sequence to its latex expression
    :param char_seq: List[{char: String, pos: (left_upper, left_lower, right_upper, right_lower)}]
    :return: String
    """
    # todo: for now ignore pos
    chars = map(lambda item: item['char'], char_seq)
    return reduce(lambda prev, cur: prev + cur, chars, '')

if __name__ == '__main__':
    char_seq = [{'char': 'x', 'pos': (0, 0)}, {'char': '=', 'pos': (0, 0)}, {'char': '5', 'pos': (0, 0)}]
    print char_seq_to_latex(char_seq)
