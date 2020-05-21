"""Main program file."""
from service import *


def main():
    """
    Function takes from the command line number of functions, start x count point
    and end x point and accurate(optional). It draws values table and graph for each
    of given functions in given file(if it`s specified) or in standard stdout.
    """
    options, arguments = std_input()
    value_table, grahps = cr_arrays(functions=arguments, boundaries=options.boundaries, accurate=options.accurate,
                                    no_frame=options.no_frame)
    output(value_table=value_table, grahps=grahps, indent=options.indent, no_tabel=options.no_tabel,
           force=options.force)


if __name__ == '__main__':
    main()
