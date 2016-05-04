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

    def insert(self, key, new_tex_let):
        """
        Insert into this TexLet
        :param key: the name of insertion point
        :param new_tex_let: the TexLet to be inserted
        """
        if key not in self.names:
            raise RuntimeError('Insertion point name not found')
        else:
            self.tex_lets[key] = new_tex_let

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
    x = SimpleTexLet('x', ())
    five = SimpleTexLet('y', ())
    x_over_five = FractionTexLet(x, five, ())
    print x_over_five