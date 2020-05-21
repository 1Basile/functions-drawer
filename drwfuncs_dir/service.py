"""Module contains main functions."""
from imports import *


def std_input() -> (tuple, dict):
    """
    Function check syntax, than read it and specified requirements.
    """
    script_name = os.path.basename(sys.argv[0])
    if "." in script_name:
        script_name = script_name.split(".")[0]
    usage = "usage: %prog [options] functions...\nfunctions - number of functions separated by space, covered with quotes."
    epilog = f"Using example: {script_name} --bondaries=[-1,1] \"y=sin(x)\" \"cos(x)\""
    description = "Script draw graphs of given functions."
    parser = OptionParser(usage=usage, version="%prog 1.0", epilog=epilog, prog=script_name, description=description)
    parser.add_option("-b", "--boundaries", dest="boundaries", type="str", help="Define x values range. Example: [-1,5]",
                      default='(0, 14)')
    parser.add_option("-a", "--accurate", dest="accurate", type="int", help="Number of points. Example: 100", default=70)
    parser.add_option('-i', '--indent', dest="indent", type='int', help="Set indent between graphs.", default=15)
    parser.add_option('-t', '--no-table', action="store_true", dest="no_tabel", help="Remove values table.",
                      default=False)
    parser.add_option('-f', '--no-frame', action="store_true", dest="no_frame", help="Remove chart frame.",
                      default=False)
    parser.add_option('-F', '--force', action="store_true", dest="force",
                      help="Return all in one block.\nNo to look at teminal window size.", default=False)
    options, arguments = parser.parse_args()

    if len(parser.parse_args()[1]) == 0:
        parser.error("no function was given.")

    if os.get_terminal_size().columns < options.accurate:
        parser.error("It`s impossible to print graphs of given size in such small terminal window.")

    if options.accurate <= 12:
        parser.error('The number of points to be drawn for function(s) cannot be less than 12.')

    try:
        options.boundaries = tuple(float(num) for num in options.boundaries[1:-1].split(','))
    except ValueError:
        parser.error('Exact boundaries was given wrong.')

    if len(options.boundaries) != 2:
        parser.error('Exact boundaries was given wrong.')

    if options.indent < 0:
        parser.error('Indent between graphs can not be less than zero.')

    for i in range(len(arguments)):
        arg = arguments[i]
        if arg.startswith('\'') or arg.startswith('\"'):
            arg = arg[1:-1]
        if arg.startswith('y='):
            arg = arg[2:]
        elif arg.startswith('y = '):
            arg = arg[4:]
        elif arg.startswith('f(x)='):
            arg = arg[5:]
        elif arg.startswith('f(x) = '):
            arg = arg[7:]
        arguments[i] = arg
    return options, arguments


def cr_arrays(functions: [str, ...], boundaries: tuple, accurate: int, no_frame: bool) \
        -> (np.array, [np.array, ...]):
    """
    Function takes list of functions and options. It creates np.array(that contains representation)
    for value_table and each graph. Return np.array of value table and list of np.arrays for graphs.
    """
    values_dict = calc_values(functions, boundaries, accurate)
    graphs_list = []
    values_table = cr_values_table(functions, values_dict)
    for function in functions:
        empty_graph = np.array([' ' for _ in range(accurate ** 2)]).reshape(accurate, accurate)
        graph = spot_points(graph=empty_graph, function=function, values_dict=values_dict, size=accurate)
        graph = add_axis(graph=graph, values_dict=values_dict, function=function, size=accurate, no_frame=no_frame)
        graphs_list.append(np.flipud(graph.T))
    return values_table, graphs_list


def calc_values(functions: list, boundaries: tuple, accurate: int) -> '{x_i: {f_1: f_1(x_i), f_2: f_2(x_i), ...}, ...}':
    """
    Function takes list of functions and calculate value of x_i for each function in given boundaries
    with point number = accurate.
    Then form dict out of its calculations. If it`s impossible to make some calculations, raise MathDoingError.

    >>> calc_values(['y=arcsin(x)', 'f(x)=x^2', 'x+2'], boundaries=(0, 5), accurate=2)
    {0.0: {'arcsin(x)': 0.0, 'x^2': 0.0, 'x+2': 2.0}, 3.0: {'arcsin(x)': 'NaN', 'x^2': 9.0, 'x+2': 5.0}}
    """
    table = {x_i: {} for x_i in np.linspace(*boundaries, accurate)}
    for function in functions:
        if function.startswith('y='):
            function = function[2:]
        if function.startswith('f(x)='):
            function = function[5:]
        try:
            func = Function(function)
        except AssertionError:
            usage = "usage: %prog [options] [function...]"
            parser = OptionParser(usage=usage, version="%prog 1.0")
            parser.error("Some functions were written wrong.")
        for x_i in table.keys():
            try:
                table[x_i].update({function: func.counter(x_i)})
            except (ValueError, AssertionError):
                table[x_i].update({function: float('nan')})

    return table


def spot_points(graph: np.array, function: str, values_dict: dict, size: int) -> np.array:
    """
    Function takes graph canvas(:empty_graph), values dict and function.
    It adds point(f_j(x_i)) onto graph for each x_i in x_array, and return modified graph.
    """
    y_values = sorted(y_list[function] for y_list in values_dict.values())
    i = 0
    j = -1
    if not all(np.isnan(y_values)):
        while np.isnan(y_values[i]):
            i += 1  # f(min) is not NaN

        while np.isnan(y_values[j]):  # f(max) is not NaN
            j -= 1
        max_y_difference = y_values[j] - y_values[i] + (y_values[j] == y_values[i])  # to avoid zero ratio
        row = 0
        for x_i in values_dict.keys():
            y_i = values_dict[x_i][function]
            if not np.isnan(y_i):
                cur_y1_value_and_min_difference = (y_i - y_values[i]) + (y_i == y_values[i]) * (
                        y_values[i] == y_values[j])
                pos_of_y = int(cur_y1_value_and_min_difference * size / max_y_difference)
                if 0 < pos_of_y < size:
                    graph[row][pos_of_y] = '*'
            row += 1
    return graph


def add_axis(graph: np.array, values_dict: dict, function: str, no_frame: bool, size: int) -> np.array:
    """Function takes graph its size, values dict and function and add x and y axis. and normalize graph."""
    y_values = sorted(list(y_list[function] for y_list in values_dict.values()))
    x_values = list(values_dict.keys())
    if no_frame:
        frame = ' '
    else:
        frame = '\u2665'

    if not all([np.isnan(y_i) for y_i in y_values]):     # if all y_values is nan
        i = 0
        j = -1
        while np.isnan(y_values[i]):
            i += 1  # f(min) is not NaN

        while np.isnan(y_values[j]):  # f(max) is not NaN
            j -= 1

        max_y_difference = y_values[j] - y_values[i] + (y_values[j] == y_values[i])  # to avoid zero ratio
        ratio_y = max_y_difference / size

        if y_values[i] < 0 < y_values[j] and round(y_values[i]) != 0:  # define x_axis location
            axis_x_pos = abs(int((- y_values[i]) / ratio_y)) + 1
            if axis_x_pos == 1:
                axis_x_pos += 1
        elif (y_values[i] < 0 or round(y_values[i] / ratio_y) == 0) and y_values[j] < 0:
            axis_x_pos = size - 1
        else:
            axis_x_pos = 2
    else:
        axis_x_pos = graph.shape[0] // 2

    max_x_difference = x_values[-1] - x_values[0] + (x_values[0] == x_values[-1])
    ratio_x = max_x_difference / size

    if x_values[0] < 0 < x_values[-1] and round(x_values[0] / ratio_x) != 0:
        axis_y_pos = abs(int((- x_values[0]) / ratio_x))
    elif x_values[0] < 0 and (x_values[-1] < 0 or round(x_values[-1] / ratio_x) == 0):  # define y_axis location
        axis_y_pos = size - 1
    else:
        axis_y_pos = 2

    up_horiz_line = list((f' {frame}' * (graph.shape[0] // 2)))
    bottom_horiz_line = list((f'{frame} ' * (graph.shape[0] // 2)))

    if size % 2 != 0:
        up_horiz_line.append(" ")
        bottom_horiz_line.append(f"{frame}")

    graph = np.c_[np.array(bottom_horiz_line).reshape(graph.shape[0], 1), graph,
                  np.array(up_horiz_line).reshape(graph.shape[0], 1)]  # make frame
    graph = np.r_[np.array([f"{frame}" for _ in range(graph.shape[1])]).reshape(1, graph.shape[1]), graph,
                  np.array([f"{frame}" for _ in range(graph.shape[1])]).reshape(1, graph.shape[1])]

    y_axis = '{2}{0:|>{1}s}\u2191{2}'.format('', graph.shape[0] - 3, frame)
    x_axis = '{2}{0:\u2015>{1}s}>{2}'.format('', graph.shape[1] - 3, frame)

    for k in range(len(x_axis)):
        if graph[k][axis_x_pos] != '*':
            graph[k][axis_x_pos] = x_axis[k]
        if graph[axis_y_pos][k] != '*':
            graph[axis_y_pos][k] = y_axis[k]
        if graph[axis_y_pos][k] != '|':
            graph[axis_y_pos][axis_x_pos] = '\u253C'

    ready_graph = set_lables(graph, axis_y_pos, axis_x_pos, y_values, x_values)

    # add header
    ready_graph = np.c_[ready_graph,
                        np.array(list(f"{f'f(x) = {function}':^{graph.shape[0]}}")).reshape(ready_graph.shape[0], 1)]

    return ready_graph


def set_lables(graph: np.array, axis_y_pos, axis_x_pos, y_values, x_values) -> np.array:
    """Function take graph, axis positions and x,y values lists. It adds labels onto graph in depends on where axis
    are located."""
    i = 0
    j = -1
    number_of_nan_from_end = 0
    number_of_nan_from_start = 0
    if not all([np.isnan(y_i) for y_i in y_values]):  # if all y_values is nan
        while np.isnan(y_values[i]):
            i += 1  # f(min) is not NaN
            number_of_nan_from_end += 1

        while np.isnan(y_values[j]):  # f(max) is not NaN
            j -= 1
            number_of_nan_from_start += 1

    upper_y = y_values[j]
    bottom_y = y_values[i]

    max_lable_lenght = max([len(f'{value:.3}') for value in [bottom_y, upper_y]])

    if (max_lable_lenght < axis_y_pos < graph.shape[0] // 2) or (graph.shape[0] - axis_y_pos - 2 < max_lable_lenght
                                                                 and axis_y_pos > graph.shape[0] // 2):
        if bottom_y != 0:
            for k in range(len(f'{bottom_y:.3}')):
                if axis_x_pos == 2:
                    graph[axis_y_pos - len(f'{bottom_y:.3}') + k][0 + 4] = f'{bottom_y:.3}'[k]
                else:
                    graph[axis_y_pos - len(f'{bottom_y:.3}') + k - 1][2] = f'{bottom_y:.3}'[k]
        else:
            for k in range(len(f'{bottom_y:.3}')):
                if axis_x_pos == 2:
                    graph[axis_y_pos - len(f'{bottom_y:.3}') + k - 1][number_of_nan_from_end + 3] = f'{bottom_y:.3}'[k]
                else:
                    graph[axis_y_pos - len(f'{bottom_y:.3}') + k - 1][number_of_nan_from_end + 1] = f'{bottom_y:.3}'[k]

        for k in range(len(f'{upper_y:.3}')):
            if axis_x_pos == graph.shape[0] - 3:
                graph[axis_y_pos - len(f'{upper_y:.3}') + k][-6] = f'{upper_y:.3}'[k]
            else:
                graph[axis_y_pos - len(f'{upper_y:.3}') + k][-4] = f'{upper_y:.3}'[k]

        graph[axis_y_pos - 2][graph.shape[1] - 2] = 'y'

    else:
        if bottom_y != 0:
            for k in range(len(f'{bottom_y:.3}')):
                if axis_x_pos == 2:
                    graph[axis_y_pos + 1 + k][number_of_nan_from_end + 4] = f'{bottom_y:.3}'[k]
                else:
                    graph[axis_y_pos + 1 + k][number_of_nan_from_end + 1] = f'{bottom_y:.3}'[k]
        if upper_y != 0:
            for k in range(len(f'{upper_y:.3}')):
                if axis_x_pos == graph.shape[0] - 3:
                    graph[axis_y_pos + 1 + k][-number_of_nan_from_start - 5] = f'{upper_y:.3}'[k]
                else:
                    graph[axis_y_pos + 1 + k][-number_of_nan_from_start - 3] = f'{upper_y:.3}'[k]
        graph[axis_y_pos + 2][graph.shape[1] - 2] = 'y'

    if x_values[0] > 1:
        for k in range(len(f'{x_values[0]:.3}')):  # inserting of boundaries points
            graph[4 + k][axis_x_pos - 1] = f'{x_values[0]:.3}'[k]

        for k in range(len(f'{x_values[-1]:.3}')):
            graph[graph.shape[0] - len(f'{x_values[-1]:.3}') - 2 + k][axis_x_pos - 1] = f'{x_values[-1]:.3}'[k]
            graph[graph.shape[0] - 3][axis_x_pos + 1] = 'x'

    elif x_values[-1] <= -1:
        for k in range(len(f'{x_values[0]:.3}')):  # inserting of boundaries points
            graph[1 + k][axis_x_pos - 1] = f'{x_values[0]:.3}'[k]

        for k in range(len(f'{x_values[-1]:.3}')):
            graph[graph.shape[0] - len(f'{x_values[-1]:.3}') - 5 + k][axis_x_pos - 1] = f'{x_values[-1]:.3}'[k]
            graph[graph.shape[0] - 2][axis_x_pos + 1] = 'x'

    elif axis_y_pos == graph.shape[0] - 3:
        for k in range(len(f'{x_values[0]:.3}')):  # inserting of boundaries points
            graph[1 + k][axis_x_pos - 1] = f'{x_values[0]:.3}'[k]

        for k in range(len(f'{x_values[-1]:.3}')):
            graph[graph.shape[0] - len(f'{x_values[-1]:.3}') - 4 + k][axis_x_pos - 1] = f'{x_values[-1]:.3}'[k]
        graph[axis_y_pos + 1][axis_x_pos - 1] = '0'
        graph[graph.shape[0] - 2][axis_x_pos + 1] = 'x'

    elif axis_y_pos in (1, 2, 3):

        for k in range(len(f'{x_values[-1]:.3}')):
            graph[graph.shape[0] - len(f'{x_values[-1]:.3}') - 2 + k][axis_x_pos - 1] = f'{x_values[-1]:.3}'[k]
        graph[axis_y_pos + 1][axis_x_pos - 1] = '0'
        graph[graph.shape[0] - 3][axis_x_pos + 1] = 'x'
    else:
        for k in range(len(f'{x_values[0]:.3}')):  # inserting of boundaries points
            graph[1 + k][axis_x_pos - 1] = f'{x_values[0]:.3}'[k]

        for k in range(len(f'{x_values[-1]:.3}')):
            graph[graph.shape[0] - len(f'{x_values[-1]:.3}') - 2 + k][axis_x_pos - 1] = f'{x_values[-1]:.3}'[k]
        graph[axis_y_pos + 1][axis_x_pos - 1] = '0'
        graph[graph.shape[0] - 3][axis_x_pos + 1] = 'x'

    return graph


def cr_values_table(functions: list, values_dict: dict) -> np.array:
    """
    Function create np.array that contains values table and return it.
    If no_tabel is True return empty array.

    >>> cr_values_table(['y=arcsin(x)', 'f(x)=x^2'], {-14.0: {'arcsin(x)': float('NaN'), 'x^2': 196.0}, 0.0: {'arcsin(x)': 0.0, 'x^2': 0.0}, 3.0: {'arcsin(x)': float('NaN'), 'x^2': 9.0}})
    [['  x   |' '  y=arcsin(x)  |' '   f(x)=x^2    |']
    ['-14.00|' '      nan      |' '      196      |']
    [' 0.00 |' '       0       |' '       0       |']
    [' 3.00 |' '      nan      |' '       9       |']]
    """
    max_x_len = max([len(f'{x_i:.2f}') for x_i in list(values_dict.keys())])  # define size of largest function
    max_row_length = sorted([len(func) for func in functions], reverse=True)[0] + 4
    for y_list in [list(y_dict.values()) for y_dict in values_dict.values()]:  # and set appropriate size of row
        max_length = max([len(f'{y_i:g}') for y_i in y_list]) + 4
        if max_length > max_row_length:  # in case, if some value is larger than functions
            max_row_length = max_length

    header = [f'{func:^{max_row_length}s}' + '|' for func in functions]
    header.insert(0, f'{"x":^{max_x_len}s}|')
    array = np.array([" "*len(i) for i in header]).reshape(1, len(header))
    array = np.vstack([array, [i for i in header]])
    for x_i in values_dict.keys():
        values = [f'{f"{y_i:g}":^{max_row_length}}' + '|' for y_i in values_dict[x_i].values()]
        x_i_str = f"{f'{x_i:.2f}':^{max_x_len}}"
        values.insert(0, f'{f"{x_i_str}":^4s}' + '|')
        array = np.vstack([array, values])
    return array


def div_output(graphs: list, indent: int, force=False):
    """Function divide graphs list onto blocks, the way they can be printed in one line, without impose.
    If Terminal window is too small it raise err."""
    column_size = int(os.get_terminal_size().columns) - 1
    blocks = []
    block = []
    block_len = 0
    i = 0
    # force to add all graphs to one block
    if force:
        blocks.append(graphs)
    else:
        for graph in graphs:
            block_len += sum(len(list(f'{elem}')) for elem in graph[0]) + indent
            if i == len(graphs) - 1:
                block_len -= indent
            if block_len <= column_size:
                block.append(graph)
            else:
                blocks.append(block)
                block = [graph]
                block_len = sum(len(list(f'{elem}')) for elem in graph[0]) + indent
            i += 1
        blocks.append(block)
    return blocks


def output(value_table: np.array, grahps: list, indent: int, no_tabel: bool, force: bool):
    """Function take value_table and list of graphs, and return it line by line in stdout."""
    indent_str = ' ' * indent
    objects_list = grahps
    if not no_tabel:
        objects_list.insert(0, value_table)
    # divide output onto part that fit terminal windows
    gropped = div_output(objects_list, indent=indent, force=force)
    for group in gropped:
        # find the longes line
        max_rows = max([np.shape(array)[0] for array in group])

        for i in range(max_rows):
            k = 0
            for object_ in group:
                if k == len(group) - 1:
                    indent_str = ''
                if i < np.shape(object_)[0]:
                    print(*object_[i], end=indent_str, sep='')
                else:
                    empty_str = len(''.join(object_[2]))
                    print(' ' * empty_str, end=indent_str, sep='')
                indent_str = ' ' * indent
                k += 1
            print("")
        print("")
