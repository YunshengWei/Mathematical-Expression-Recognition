class TexLet(object):
    """
    A TexLet is a string of Latex with custom named insertion points
    TexLets are inserted into the template for recursive LaTex construction
    """
    def __init__(self, template, insertion_points, extra):
        """
        Construct a TexLet
        :param template: String, the LaTex template
        :param insertion_points: {String: Number}, custom insertion points
                key of a key-value-pair is the name of the insertion point
                value of a key-value-pair is the position that insertion will happen after
        :param extra: extra information
        """
        self.template = template
        import operator
        sorted_insertion_points = sorted(insertion_points.items(), key=operator.itemgetter(1))
        self.names = map(lambda i: i[0], sorted_insertion_points)
        self.poses = map(lambda i: i[1], sorted_insertion_points)
        self.tex_lets = {}
        self.extra = extra

    def insert(self, name, new_tex_let):
        """
        Insert into this TexLet
        :param name: the name of insertion point
        :param new_tex_let: the TexLet to be inserted
        """
        if name not in self.names:
            raise RuntimeError('Insertion point name not found')
        else:
            self.tex_lets[name] = new_tex_let

    def __str__(self):
        """
        Build the LaTex recursively, expanding the TaxLet's on the insertion points
        :return:
        """
        if len(self.names) > 0:
            first_pos = self.poses[0]
            last_pos = self.poses[len(self.poses) - 1]

            def index_to_tex_let(index):
                return self.tex_lets[self.names[index]]

            latex = self.template[:first_pos + 1] + str(index_to_tex_let(0))
            for i in range(1, len(self.poses)):
                pos = self.poses[i]
                prev_pos = self.poses[i - 1]
                latex += self.template[prev_pos + 1: pos + 1] + str(index_to_tex_let(i))
            latex += self.template[last_pos + 1:]

            return latex
        else:
            return self.template


class SimpleTexLet(TexLet):
    def __init__(self, text, extra):
        TexLet.__init__(self, template=text, insertion_points={}, extra=extra)


class FractionTexLet(TexLet):
    def __init__(self, numerator, denominator, extra):
        TexLet.__init__(self,
                        template='\\frac{}{}',
                        insertion_points={'numerator': 5, 'denominator': 7},
                        extra=extra)
        self.insert('numerator', numerator)
        self.insert('denominator', denominator)


def horizontal_positioning(input_tex_lets):
    """
    :param input_tex_lets: [TexLet]
    :return: [TexLet]
    """
    if len(input_tex_lets) == 0:
        return SimpleTexLet('', {})
    else:
        res = reduce(lambda prev, cur: str(prev) + str(cur), input_tex_lets, SimpleTexLet('', {}))
        return SimpleTexLet(res, {})


FRACTION = '\\frac'


def fraction_positioning(input_tex_lets):
    """
    :param input_tex_lets: [TexLet]
    :return: [TexLet]
    """
    fractions = filter(lambda l: l.template == FRACTION, input_tex_lets)

    if len(fractions) != 0:
        # find the longest fraction TexLet
        def longer_fraction(x, y):
            return x.extra['right'] - x.extra['left'] > y.extra['right'] - y.extra['left']
        fractions.sort(longer_fraction)
        fraction = fractions[len(fractions) - 1]

        # find and evaluate TexLets to left of that
        def to_left(t):
            return t.extra['right'] < fraction.extra['left']
        left = heuristic_evaluate(filter(to_left, input_tex_lets))

        # find and evaluate TexLets to right of that
        def to_right(t):
            return fraction.extra['right'] < t.extra['left']
        right = heuristic_evaluate(filter(to_right, input_tex_lets))

        # find and evaluate TexLets above that
        def inbound(t):
            return not to_left(t) and not to_right(t)

        def to_above(t):
            return inbound(t) and t.extra['lower'] <= fraction.extra['upper']
        above = heuristic_evaluate(filter(to_above, input_tex_lets))

        # find and evaluate TexLets below that
        def to_below(t):
            return inbound(t) and fraction.extra['lower'] <= t.extra['upper']
        below = heuristic_evaluate(filter(to_below, input_tex_lets))

        # Construct
        return [left, FractionTexLet(above, below, fraction.extra), right]
    else:
        return input_tex_lets


def heuristic_evaluate(tex_lets):
    """
    :param tex_lets: [TexLet]
    :return: TexLet
    """
    from_fraction = fraction_positioning(tex_lets)
    from_horizontal = horizontal_positioning(from_fraction)
    return from_horizontal


def char_seq_to_latex(char_seq):
    """
    Converts a char sequence to its latex expression
    :param char_seq: List[{char: String, pos: {upper: Number, lower: Number, left: Number, right: Number}}]
    :return: String
    """
    tex_lets = map(lambda char: SimpleTexLet(char['char'], char['pos']), char_seq)
    res_tex_let = heuristic_evaluate(tex_lets)
    return str(res_tex_let)

if __name__ == '__main__':
    x = {'char': 'x', 'pos': {'upper': 0, 'lower': 1, 'left': 0, 'right': 1}}
    over = {'char': '\\frac', 'pos': {'upper': 1.5, 'lower': 2, 'left': 0, 'right': 1}}
    five = {'char': '5', 'pos': {'upper': 2.5, 'lower': 3, 'left': 0, 'right': 1}}
    print char_seq_to_latex([x, over, five])
