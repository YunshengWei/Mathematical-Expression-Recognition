FRACTION = '-'
SUPERSCRIPT_DIFF = 10
SQUARE_ROOT = '\\sqrt'


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


class SuperscriptTexLet(TexLet):
    def __init__(self, base, super, extra):
        TexLet.__init__(self,
                        template='^{}',
                        insertion_points={'base': -1, 'super': 1},
                        extra=extra)
        self.insert('base', base)
        self.insert('super', super)


class SquareRootTexLet(TexLet):
    def __init__(self, inner, extra):
        TexLet.__init__(self,
                        template='\\sqrt{}',
                        insertion_points={'inner': 5},
                        extra=extra)
        self.insert('inner', inner)


def most_upper(tex_lets):
    if len(tex_lets) == 0:
        return -1
    return min(map(lambda t: t.extra['upper'], tex_lets))


def most_lower(tex_lets):
    if len(tex_lets) == 0:
        return -1
    return max(map(lambda t: t.extra['lower'], tex_lets))


def most_left(tex_lets):
    if len(tex_lets) == 0:
        return -1
    return min(map(lambda t: t.extra['left'], tex_lets))


def most_right(tex_lets):
    if len(tex_lets) == 0:
        return -1
    return max(map(lambda t: t.extra['right'], tex_lets))


def box(tex_lets):
    return {'lower': most_lower(tex_lets),
            'upper': most_upper(tex_lets),
            'left': most_left(tex_lets),
            'right': most_right(tex_lets)
            }


def horizontal_positioning(input_tex_lets):
    """
    :param input_tex_lets: [TexLet]
    :return: [TexLet]
    """
    res = reduce(lambda prev, cur: str(prev) + str(cur), input_tex_lets, '')
    return SimpleTexLet(res, box(input_tex_lets))


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
            return t.extra['right'] <= fraction.extra['left']
        left = heuristic_evaluate(filter(to_left, input_tex_lets))

        # find and evaluate TexLets to right of that
        def to_right(t):
            return fraction.extra['right'] <= t.extra['left']
        right = heuristic_evaluate(filter(to_right, input_tex_lets))

        # find and evaluate TexLets above that
        def inbound(t):
            return not to_left(t) and not to_right(t)

        def to_above(t):
            return inbound(t) and t.extra['lower'] <= fraction.extra['upper']
        above_lets = filter(to_above, input_tex_lets)
        new_upper = most_upper(above_lets)
        above = heuristic_evaluate(above_lets)

        # find and evaluate TexLets below that
        def to_below(t):
            return inbound(t) and fraction.extra['lower'] <= t.extra['upper']
        below_lets = filter(to_below, input_tex_lets)
        new_lower = most_lower(below_lets)
        below = heuristic_evaluate(below_lets)

        # Construct
        fraction.extra['upper'] = new_upper
        fraction.extra['lower'] = new_lower
        return [left, FractionTexLet(above, below, fraction.extra), right]
    else:
        return input_tex_lets


def superscript_positioning(input_tex_lets):
    """
    :param input_tex_lets: [TexLet]
    :return: [TexLet]
    """
    # find center lines
    centers = map(lambda t: t.extra['upper'] + (t.extra['lower'] - t.extra['upper']) / float(2), input_tex_lets)

    if len(centers) > 0:
        # try to find a hill
        base_center = centers[0]
        base_end = len(centers) - 1
        super_center = 1
        finding_super = False

        for index, pos in enumerate(centers):
            if finding_super:
                if super_center - pos >= SUPERSCRIPT_DIFF:
                    # base
                    base_input_lets = input_tex_lets[:base_end + 1]
                    base = heuristic_evaluate(base_input_lets)

                    # super
                    super_input_lets = input_tex_lets[base_end + 1: index]
                    super = heuristic_evaluate(super_input_lets)

                    # the rest
                    the_rest = heuristic_evaluate(input_tex_lets[index:])

                    # construct
                    new_box = box(base_input_lets + super_input_lets)
                    return [SuperscriptTexLet(base, super, new_box)] + the_rest
            else:
                if base_center - pos >= SUPERSCRIPT_DIFF:
                    base_end = index - 1
                    super_center = pos
                    finding_super = True

        if not finding_super:
            return input_tex_lets
        else:
            # base
            base_input_lets = input_tex_lets[:base_end + 1]
            base = heuristic_evaluate(base_input_lets)

            # super
            super_input_lets = input_tex_lets[base_end + 1:]
            super = heuristic_evaluate(super_input_lets)

            # construct
            return [SuperscriptTexLet(base, super, box(input_tex_lets))]

    else:
        return input_tex_lets


def square_root_positioning(input_tex_lets):
    """
    :param input_tex_lets: [TexLet]
    :return: [TexLet]
    """
    square_roots = filter(lambda l: l.template == SQUARE_ROOT, input_tex_lets)

    if len(square_roots) != 0:
        # find biggest square root
        def bigger_square_root(x, y):
            return x.extra['upper'] < y.extra['upper'] and y.extra['lower'] < x.extra['lower'] \
                   and x.extra['left'] < y.extra['left'] and y.extra['right'] < x.extra['right']
        square_roots.sort(bigger_square_root)
        square_root = square_roots[len(square_roots) - 1]

        # find and evaluate TexLets to left of that
        def to_left(t):
            return t.extra['right'] <= square_root.extra['left']
        left = heuristic_evaluate(filter(to_left, input_tex_lets))

        # find and evaluate TexLets to right of that
        def to_right(t):
            return square_root.extra['right'] <= t.extra['left']
        right = heuristic_evaluate(filter(to_right, input_tex_lets))

        # find and evaluate TexLets inside that
        def inbound(t):
            return not to_left(t) and not to_right(t) and t.template != SQUARE_ROOT
        inner = heuristic_evaluate(filter(inbound, input_tex_lets))

        # Construct
        return [left, SquareRootTexLet(inner, square_root.extra), right]
    else:
        return input_tex_lets


def heuristic_evaluate(tex_lets):
    """
    :param tex_lets: [TexLet]
    :return: TexLet
    """
    from_fraction = fraction_positioning(tex_lets)
    from_superscript = superscript_positioning(from_fraction)
    from_square_root = square_root_positioning(from_superscript)
    from_horizontal = horizontal_positioning(from_square_root)
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
    one = {'char': '1', 'pos': {'upper': 1.5, 'lower': 2, 'left': 0, 'right': 0.5}}
    plus = {'char': '+', 'pos': {'upper': 1.5, 'lower': 2, 'left': 0.5, 'right': 1}}
    x = {'char': 'x', 'pos': {'upper': 0, 'lower': 1, 'left': 1, 'right': 2}}
    over = {'char': '-', 'pos': {'upper': 1.5, 'lower': 2, 'left': 1, 'right': 2}}
    five = {'char': '5', 'pos': {'upper': 2.5, 'lower': 3, 'left': 1, 'right': 2}}
    minus = {'char': '-', 'pos': {'upper': 1.5, 'lower': 2, 'left': 2, 'right': 2.5}}
    y = {'char': 'y', 'pos': {'upper': 1.5, 'lower': 2, 'left': 2.5, 'right': 3}}
    print char_seq_to_latex([one, plus, x, over, five, minus, y])

    a = {'char': 'a', 'pos': {'upper': 20, 'lower': 21, 'left': 0, 'right': 1}}
    b = {'char': 'b', 'pos': {'upper': 0, 'lower': 1, 'left': 1, 'right': 1.5}}
    plus = {'char': '+', 'pos': {'upper': 0, 'lower': 1, 'left': 1.5, 'right': 2}}
    c = {'char': 'c', 'pos': {'upper': 0, 'lower': 1, 'left': 2, 'right': 2.5}}
    print char_seq_to_latex([a, b, plus, c])

    sqrt = {'char': '\\sqrt', 'pos': {'upper': 0, 'lower': 2, 'left': 0, 'right': 5}}
    p = {'char': 'p', 'pos': {'upper': 1, 'lower': 1.5, 'left': 1, 'right': 2}}
    q = {'char': 'q', 'pos': {'upper': 1, 'lower': 1.5, 'left': 3, 'right': 4}}
    print char_seq_to_latex([sqrt, p, q])




