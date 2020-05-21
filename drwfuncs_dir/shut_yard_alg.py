"""
Shut Yard Algorithm
=====


This is an object-oriented plotting library.

Provides:
    1) Class Function is a shell for work with functions.
    2) Class Stack is a collection of objects. -> LIFO data structure.
    3) Class Queue is a collection of objects. -> FIFO data structure.

How to use the documentation
----------------------------

The docstring examples assume that `shut_yard_alg` has been imported as `*`::

  >>> from shut_yard_alg import *
"""
import math


class _Node:
    """
    Class contains reference to data value, and reference to the next Node.
    """

    def __init__(self, data):
        self.data = data
        self.next = None


class MathDomainError(ArithmeticError, ValueError):
    """Math doing err"""
    pass


class Stack:
    """
    Class Stack is a collection of objects.
    LIFO data structure.
    Stack contains instances of Node.
    It only points to the current head element(of type Node).
    Support standard set of operations: insert(push), remove(pop), iterate over
    values, check size/ if empty(is_empty).
    """

    def __init__(self):
        self.head = None

    def is_empty(self):
        """
        Method check if stack is empty.
        """
        return self.head is None

    def push(self, data):
        """
        Method create new Node that points to the current head(since the new Node is
        on top of the stack), and reassign the head reference to this new Node.
        """
        new_node = _Node(data)
        new_node.next = self.head
        self.head = new_node

    def pop(self):
        """
        Method remove Node from the top of the stack, save the Node we are removing,
        reassign the head to point to next element, and return data contained in the Node we remove.
        """
        if self.is_empty():
            return None
        popped_node = self.head
        self.head = self.head.next
        popped_node.next = None
        return popped_node.data

    def __str__(self):
        if self.is_empty():
            return '[]'

        current_node = self.head
        string = ''
        while current_node is not None:
            if string:
                string += ', '
            else:
                string += '['
            string += str(current_node.data)
            current_node = current_node.next
        string += ']'
        return string

    def __len__(self):
        length = 0
        current_node = self.head
        while current_node is not None:
            length += 1
            current_node = current_node.next
        return length

    class StackIterator:
        """
        It iterate over Stack.
        """

        def __init__(self, head):
            self.current_node = head

        def __next__(self):
            if self.current_node is None:
                raise StopIteration

            value = self.current_node.data
            self.current_node = self.current_node.next
            return value

    def __iter__(self):
        return self.StackIterator(self.head)


class Queue:
    """
    Class Queue is a collection of objects.
    FIFO data structure.
    Queue contains instances of Node.
    It points to the current head and tail elements(of type Node).
    Support standard set of operations: insert(enqueue), remove(dequeue), iterate over,
    values, check size/ if empty(is_empty).
    """

    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        """
        Method check if queue is empty.
        """
        return self.head is None

    def enqueue(self, data):
        """
        Method create new Node that points to the current tail(since the new Node is
        on bottom of the stack), and reassign the tail reference to this new Node.
        """
        old_tail = self.tail
        self.tail = _Node(data)
        if self.is_empty():
            self.head = self.tail
        else:
            old_tail.next = self.tail

    def dequeue(self):
        """
        Method remove Node from the top of the stack, save the Node we are removing,
        reassign the head to point to next element, and return data contained in the Node we remove.
        """
        if self.is_empty():
            return None
        head_node = self.head
        self.head = self.head.next
        if self.is_empty():
            self.tail = None
        return head_node.data

    def to_list(self):
        """
        Method transform Queue onto List.
        """
        return [data for data in self]

    def __str__(self):
        if self.head is None:
            return '{}'

        current_node = self.head
        string = ''
        while current_node is not None:
            if string:
                string += ', '
            else:
                string += '{'
            string += str(current_node.data)
            current_node = current_node.next
        string += '}'
        return string

    def __len__(self):
        length = 0
        current_node = self.head
        while current_node is not None:
            length += 1
            current_node = current_node.next
        return length

    class QueueIterator:
        """
        It iterate over Queue.
        """

        def __init__(self, head):
            self.current_node = head

        def __next__(self):
            if self.current_node is None:
                raise StopIteration

            value = self.current_node.data
            self.current_node = self.current_node.next
            return value

    def __iter__(self):
        return self.QueueIterator(self.head)


class Function(object):
    """
    Class works with string written expression(functions). It can calculate them,
    divide onto tokens, divide onto reverse polish notation.
    Need queues and stacks to work.

    Notes:
        A coefficient should be written before a variable.
        A coefficient and functions(like ln(x)) should be written with '*' between them.
        If you are multiplying variable onto function in should be written after function(like cos(x)x).
        Parameter is positional arg, by default is 'x'.
        It is impossible to use as parameters: 's', 'l', 'a', 'c', 't', 'a', 'p', 'e', '@'.
        Power of log must not have operands.
    """

    def __init__(self, expr, *, param="x"):
        self.parameter = str(param)
        self.expr = expr
        self.__identifikator = self.expr
        self.__derivative = None
        self.tokens = self._token_div()
        self.rp_ord = self._rev_pol_ord()
        self.result = None

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.expr)

    def __str__(self):
        return self.__expr

    def __hash__(self):
        return hash(id(self))

    def __format__(self, format_spec):
        return format(self.__expr, format_spec)

    def __abs__(self):
        self.expr = f"abs({self.__expr})"
        return self

    def __neg__(self):
        self.expr = f"-({self.__expr})"
        return self

    for __name, __operator in (("__add__", "+"), ("__sub__", "-"), ("__rfloordiv__", "//"), ("__rtruediv__", "/"),
                               ("__mul__", "*"), ("__mod__", "%"), ("__floordiv__", "//"), ("__truediv__", "/"),
                               ("__pow__", "**"), ("__iadd__", "+"), ("__isub__", "-"), ("__ipow__", "**"),
                               ("__imul__", "*"), ("__imod__", "%"), ("__ifloordiv__", "//"), ("__itruediv__", "/"),
                               ("__radd__", "+"), ("__rsub__", "-"), ("__rpow__", "**"), ("__rmul__", "*"),
                               ("__rmod__", "%")):

        if __operator == '//':
            __rep_operator = '&'
        elif __operator == '**':
            __rep_operator = '^'
        else:
            __rep_operator = __operator

        if "i" == __name[2]:
            __command = "\t\"\"\"\n" \
                        "\tMethod change current object, does not create new one.\n" \
                        "\t>>> test1 = Function(\'x**x + x^3 + ln(x)\')\n" \
                        "\t>>> test2 = Function(\'x^(1/3) - pi\')\n" \
                        f"\t>>> test1 {__operator}= test2\n" \
                        "\t>>> test1\n" \
                        f"\tFunction((x^x+x^3+ln(x)){__rep_operator}(x^(1/3)-pi))\n" \
                        "\t\"\"\"\n" \
                        f"\tself.expr = \"(\" + self.expr + \")\" + \"{__operator}\" + \"(\" + other.expr + \")\"\n" \
                        "\treturn self"
        else:
            __command = "\t\"\"\"\n" \
                        "\tMethod return new object.\n" \
                        "\t>>> test1 = Function(\'x**x + x^3 + ln(x)\')\n" \
                        "\t>>> test2 = Function(\'x^(1/3) - pi\')\n" \
                        f"\t>>> test1 {__operator} test2\n" \
                        f"\tFunction((x^x+x^3+ln(x)){__rep_operator}(x^(1/3)-pi))\n" \
                        "\t\"\"\"\n" \
                        f"\treturn Function(\"(\" + self.expr + \")\" + \"{__operator}\" + \"(\" + other.expr + \")\")"

        exec(f'def {__name}(self, other):\n'
             f'{__command}\n')

        del __rep_operator
        del __command
        del __name
        del __operator

    @staticmethod                   # TODO: Make derivative finder
    def _space_deleter(string: str) -> str:
        """
        Method delete space from expression.

        >>> test = Function('x+ sin (x + 1)')
        >>> test._space_deleter('x+ sin (x + 1)')
        'x+sin(x+1)'
        """
        divided_expr = string.split(' ')
        expr_changed = ''.join(divided_expr)
        return expr_changed

    @staticmethod
    def _revers_str(string: str) -> str:
        """
        Method revers string.

        >>> test = Function("2**x")       # whatever expression
        >>> test._revers_str('456')
        '654'
        """
        rev_str = ''
        for i in range(len(string) - 1, -1, -1):
            rev_str += string[i]
        return rev_str

    @staticmethod
    def _is_func(token: str) -> str:
        """
        Method check if str is a full function or constant.

        Means "arcct" is not a full one. It return only full ones.

        >>> test = Function("2**x")       # whatever expression
        >>> test._is_func("arc")
        ''

        >>> test = Function("2**x")       # whatever expression
        >>> test._is_func("ln")
        'ln'
        """
        func = ['f', 'p', 'e', 's', 'l', 'a', 'c', 't', 'a', 'fa', 'si', 'cs', 'lo', 'co', 'ct', 'ar', 'se', 'ab',
                'fac', 'arc', 'fact', 'arcc', 'arcs', 'arct', 'facto', 'arcco', 'arcct', 'arcsi', 'factorial', 'sin',
                'pi', 'cos', 'tg', 'ctg', 'arccos', 'arcsin', 'arcctg', 'arctg', 'sec', 'csc', 'log', 'ln', 'abs', 'e']
        wrong_writing = ['lg', 'cosec', 'fabs']
        right_writing = ['log', 'csc', 'abs']
        for i in wrong_writing:
            while i in token:
                token = token.replace(i, right_writing[wrong_writing.index(i)])

        if token in func[:]:
            if token in func[func.index('fa'):]:
                if token in func[func.index('fac'):]:
                    if token in func[func.index('fact'):]:
                        if token in func[func.index('facto'):]:
                            if token in func[func.index('factorial'):]:
                                return token
        else:
            return token
        token = ''
        return token

    @property
    def expr(self):
        """
        Function in string cover.

        >>> test = Function("2**x")
        >>> test.expr
        '2^x'

        >>> test = Function("-5x")
        >>> test.expr
        '-1*5*x'

        >>> test = Function("5//x")
        >>> test.expr
        '5&x'

        >>> test = Function("5(3x))")
        Traceback (most recent call last):
        '''
        AssertionError: Function was written wrong.
        Total number of '(' and ')' must be the same.
        """
        return self.__expr

    @expr.setter
    def expr(self, expr):
        """
        Method change operands("**", "//" or missing "*") in expression onto ("^", "&", "*").
        Also check number of '(' and ')' in expression.

        If revers = 0 (by default), it change "//" onto "&", and "**" onto "^".
        Change "&" onto "//", and "^" onto "**" otherwise.
        Can be also used with lists.
        """
        signs = [" ", "*", "+", "-", "**", "/", "%", "//", "^", "&"]
        to_change = ["//", "**", "^", "&"]
        param = self.parameter
        expr_changed = self._space_deleter(expr)

        assert expr_changed.count("(") == expr_changed.count(")"), \
            """Function was written wrong.\nTotal number of '(' and ')' must be the same."""

        for i in to_change[:2]:
            while i in expr_changed:
                tmp = ""
                a = expr_changed.split(i, maxsplit=1)
                a[0] += to_change[len(to_change) - (to_change.index(i) + 1)]
                for k in a:
                    tmp += k
                expr_changed = tmp

        while param in expr_changed:
            indx = expr_changed.index(param)
            if indx == 0:
                expr_changed = expr_changed.replace(param, "@", 1)
            else:
                tmp = ""
                if expr_changed[indx - 1] not in signs:
                    a = expr_changed.split(expr_changed[indx], maxsplit=1)
                    a[0] += "*@"
                    for i in a:
                        tmp += i
                    expr_changed = tmp
                else:
                    expr_changed = expr_changed.replace(param, "@", 1)
        if "@" in expr_changed:
            expr_changed = expr_changed.replace("@", param)
        while '(' in expr_changed:
            tmp = expr_changed.index('(') + 1
            if expr_changed[tmp] is "*":
                expr_changed = expr_changed[:tmp] + expr_changed[tmp:tmp + 1].replace(expr_changed[tmp], '') \
                               + expr_changed[tmp + 1:]
            expr_changed = expr_changed.replace("(", "@", 1)
        if "@" in expr_changed:
            expr_changed = expr_changed.replace("@", "(")
        while "-" in expr_changed:
            indx = expr_changed.index("-")  # change unary minus to '-1*'
            if (indx == 0) or ((expr_changed[indx - 1] in signs) or (expr_changed[indx - 1] == '(')):
                expr_changed = expr_changed[0:indx] + "@1*" + expr_changed[indx + 1:]
            else:
                expr_changed = expr_changed.replace("-", "@", 1)
        if "@" in expr_changed:
            expr_changed = expr_changed.replace("@", "-")
        self.__expr = expr_changed

    def _token_div(self):
        """
        Method divides string onto tokens.

        Remarks:
            "244.44", "sin", "(", "//" is one token.
            Log power should be param or number(without operators).
            It use: _is_func(), _revers_str(), _space_deleter().


        >>> test = Function("21.3 ** x")
        >>> test.tokens
        ['21.3', '**', 'x']

        >>> test = Function("y + sin(y+1)", param='y')
        >>> test.tokens
        ['y', '+', 'sin', '(', 'y', '+', '1', ')']
        """
        operat_func = ['+', '-', '/', '%', '//', '*', '**', '(', ')', 'sin', 'cos', 'tg', 'ctg', 'arcctg', 'arctg',
                       'arccos', 'arcsin', 'sec', 'csc', 'cot', 'log', 'ln', 'abs', 'factorial']
        func = []
        signs = ["*", "+", "-", "^", "/", "%", "&", "(", ")", "%", "log", "ln"]
        constants = ['pi', 'e', 'inf', '-inf', '-0', '+0', '-1']
        to_change = ["&", "^", "**", "//"]
        tokens = []
        real_indx = 0
        c = 0
        tmp = ''

        for elem in self.expr:
            right_elem = real_indx
            if elem in signs:
                tokens.append(elem)
            else:
                tmp += self.expr[right_elem]
                while self.expr[right_elem - 1] not in signs and right_elem >= 1:
                    tmp += self.expr[right_elem - 1]
                    right_elem -= 1
                tmp = self._revers_str(tmp)
                if real_indx == 0:
                    right_elem = real_indx + 1
                else:
                    right_elem = real_indx + c
                if right_elem > 1:
                    if tokens[right_elem - 1] in tmp and 1 <= right_elem <= (len(tokens)):
                        if tokens[right_elem - 1] is not tmp:
                            tokens.pop()
                            c -= 1
                check = self._is_func(tmp)
                if check:
                    tokens.append(check)
                else:
                    c -= 1
                tmp = ''

            real_indx += 1
        if len(tokens) != 1 and tokens[0] in tokens[1]:  # Реализация обработки первого 1+ цифрового числа на первой позиции
            tokens.pop(0)
        if tokens[0:2] == ['-', '1']:  # Замена унарного минуса
            tokens.pop(1)
            tokens[0] = '-1'
        for i in to_change[:2]:  # Реализация обратной замены для 'to_change'
            while i in tokens:
                tokens.insert(tokens.index(i), to_change[len(to_change) - (to_change.index(i) + 1)])
                tokens.remove(i)
        if "log{" in tokens:  # Реализация log з основанием x
            indx = tokens.index("log{")
            tokens.pop(indx + 1)
            tokens[indx] += tokens.pop(indx + 1)
        if "!" in tokens:  # Реализация факториала
            real_indx = tokens.index("!")
            if tokens[real_indx - 1] != ")":
                tokens.insert(real_indx - 1, 'factorial')
                tokens.insert(real_indx - 1, '(')
                tokens[real_indx] = ")"
                tokens.pop(tokens.index("!"))
            else:
                indx = tokens[:real_indx].index("(") + 1
                while len(tokens[:indx]) + tokens[indx:].index(")") != real_indx - 1:
                    indx = tokens[:real_indx].index("(") + 1
                    if tokens[indx:real_indx].index(")") < tokens[indx:real_indx].index("("):
                        tokens[tokens[indx - 1:real_indx].index(")")] = "$"
                        tokens[tokens[indx - 1:real_indx].index("(")] = "@"
                    else:
                        indx1 = len(tokens[:indx]) + tokens[indx:real_indx].index(")")
                        indx2 = len(tokens[:indx]) + tokens[indx:real_indx].index("(")
                        tokens[indx1] = "$"
                        tokens[indx2] = "@"
                tokens.insert(tokens.index("("), 'factorial')
                tokens.pop(tokens.index("!"))
                while "$" in tokens:
                    tokens[tokens.index("$")] = ")"
                    tokens[tokens.index("@")] = "("
        test_list = tokens[:]  # Проверка на неправильный ввод
        while self.parameter in test_list:
            test_list.pop(test_list.index(self.parameter))
        for elem in (operat_func + constants):
            while elem in test_list:
                test_list.pop(test_list.index(elem))
        # print(test_list)
        for token in test_list:
            if 'log{' in token and token[-1] == '}':
                token = token[4:-1]
                if token == 'x':
                    continue
            for elem in token:
                assert elem in "0123456789.", \
                    "Function was written wrong.\nThere is an incorrect token in given function."

        return tokens

    @property
    def derivative(self):
        """
        This method return derivative of given function.

            Remarks:
                If derivative does not exist return 0.

        Here must be right doc_tests.........
        >> test = Function('x^x + 3')
        >> test.derivative
        'x^x'
        """
        if self.__derivative is not None:
            if self.__identifikator != self.expr:  # Check if expr changed until initialization
                self.tokens = self._token_div()
                self.rp_ord = self._rev_pol_ord()
                self.expr = self.__identifikator
        else:
            param = self.parameter
            ch_expr = self.expr[:]
            rules = {"(": (lambda a: a.index(")")), }
            to_split_for = ["+", "-"]
            for elem in ch_expr:
                first_op = rules[elem](ch_expr)
                pass


    def _rev_pol_ord(self):
        """
        Method moves tokens in a correct order to do calculations.

        Remarks:
            It works with Queues and Stacks.


        >>> test = Function('y + tg(y^3)', param='y')
        >>> test.rp_ord
        ['y', 'y', '3', '**', 'tg', '+']

        >>> test = Function('z * arcctg(z//4)', param='z')

        >>> test.rp_ord
        ['z', 'z', '4', '//', 'arcctg', '*']
        """
        operat_func = {"+": 1, "-": 1, "/": 2, "%": 2, "//": 2, "*": 2, "**": 3, 'sin': 4, 'cos': 4, 'tg': 4, 'ctg': 4,
                       'arcctg': 4, 'arctg': 4, 'arccos': 4, 'arcsin': 4, 'sec': 4, 'csc': 4, 'cot': 4, 'log{}': 4,
                       'ln': 4, 'abs': 4, 'factorial': 4}
        stack = Stack()
        queue = Queue()
        k = 0

        while len(self.tokens) > k:
            token = self.tokens[k]
            k += 1
            if 'log{' in token:
                log_power = '{' + token[4:-1] + '}'
                token = token[:4] + token[-1]
            if token is "(":
                stack.push(token)
            elif token is ")":
                while stack.head.data is not "(":
                    queue.enqueue(stack.pop())  # move operators from stack to queue
                stack.pop()
            elif token in operat_func:
                if stack.is_empty():
                    stack_head = stack.head
                else:
                    stack_head = stack.head.data
                while (stack_head in operat_func) and (
                        operat_func.setdefault(stack_head) > operat_func.setdefault(token)):
                    queue.enqueue(stack.pop())
                    if stack.is_empty():
                        stack_head = stack.head
                    else:
                        stack_head = stack.head.data
                stack.push(token)
            else:
                queue.enqueue(token)
        while not stack.is_empty():
            queue.enqueue(stack.pop())
        rp_ord = queue.to_list()
        while 'log{}' in rp_ord:
            indx = rp_ord.index('log{}')
            rp_ord[indx] = f"log{log_power}"
        return rp_ord


    def counter(self, value, *, to_const=False):
        """
        This method count expression with given (param) and equates result with constants.

        Remarks:
            Method equates result with constants if 'to_const"= True.(by default= False)

        >>> test = Function('x^x + 3//x')
        >>> test.counter(3)
        28.0

        >>> test = Function('x^x + x^3 + log{x}(x)x')
        >>> test.counter(3)
        57.0
        """
        constants = {"pi": math.pi, "e": math.e, "inf": 1e+300, "-inf": -1e+300, "-0": -1e-300,
                     "+0": 1e-300}

        equal_to_const = {"inf": lambda a: (a > 1e+50), "pi": lambda a: (3.14 < a < 3.141593),
                          "e": lambda a: (2.71 < a < 2.71829), "-inf": lambda a: (a < -1e+50),  # some constants
                          "+0": lambda a: (0 < a < 1e-10), "-0": lambda a: (-1e-10 < a < 0)}

        operators_priority = {"+": 1, "-": 1, "/": 2, "%": 2, "//": 2, "*": 2, "**": 3}

        operators = {"+": lambda a, b: (a + b), "-": lambda a, b: (a - b), "/": lambda a, b: (a / b),
                     "%": lambda a, b: (a % b), "//": lambda a, b: (a // b), "*": lambda a, b: (a * b),
                     "**": lambda a, b: (a ** b)}

        functions_priority = {'sin': 4, 'cos': 4, 'tg': 4, 'ctg': 4,
                              'arcctg': 4, 'arctg': 4, 'arccos': 4, 'arcsin': 4, 'sec': 4, 'csc': 4, 'cot': 4,
                              'ln': 4, 'abs': 4, 'factorial': 4}

        functions = {"sin": lambda a: (math.sin(a)), "cos": lambda a: (math.cos(a)), "tg": lambda a: (math.tan(a)),
                     "ctg": lambda a: (1 / math.tan(a)), "arcctg": lambda a: (math.pi / 2 - math.atan(a)),
                     "arccos": lambda a: (math.acos(a)), "arcsin": lambda a: (math.asin(a)),
                     "arctg": lambda a: (math.atan(a)), "sec": lambda a: (math.cosh(a)),
                     "csc": lambda a: (math.sinh(a)), "cot": lambda a: (math.tanh(a)), "ln": lambda a: (math.log(a)),
                     "abs": lambda a: (math.fabs(a)), "log{": lambda sign, a: (math.log(a, float(sign[4:-1]))),
                     "factorial": lambda a: (math.factorial(a))}

        if self.__identifikator != self.expr:  # Check if expr changed until initialization
            self.tokens = self._token_div()
            self.rp_ord = self._rev_pol_ord()
            self.expr = self.__identifikator
        param = self.parameter
        ch_list = self.rp_ord[:]

        try:
            for element in ch_list:
                if param in element:
                    indx = ch_list.index(element)
                    ch_list[indx] = element.replace(param, str(value))  # change 'param' onto 'value'
                    element = ch_list[indx]
                if element in constants:
                    indx = ch_list.index(element)  # it change const onto it`s value
                    ch_list[indx] = str(constants[element])
            if len(ch_list) == 1:
                res = float(ch_list[0])
            else:
                while len(ch_list) != 1:
                    flag = 0  # it is using to restart iteration
                    if len(ch_list) == 2 and ch_list[1] == "-":
                        ch_list.pop(1)          # if унарый минус и число остались
                        ch_list[0] = -ch_list[0]
                        continue
                    for token in ch_list:
                        if flag == 1:
                            break
                        indx = ch_list.index(token)
                        if token in operators_priority:
                            flag = 1
                            sign = ch_list.pop(indx)
                            b = float(ch_list.pop(indx - 1))
                            a = float(ch_list.pop(indx - 2))  # a - first operand, b - second one
                            res = operators[sign](a, b)
                            ch_list.insert(indx - 2, res)
                        elif (token in functions_priority) or (
                                str(token)[:4] == "log{"):
                            flag = 1
                            sign = ch_list.pop(indx)  # function
                            a = float(ch_list.pop(indx - 1))  # a - operand
                            if "log{" in sign:
                                res = functions["log{"](sign, a)
                            else:
                                res = functions[sign](a)
                            ch_list.insert(indx - 1, res)
                res = ch_list[0]

                if to_const:
                    for i, const in enumerate(list(equal_to_const.values())):
                        is_const = const(res)
                        if is_const:
                            res = list(equal_to_const.keys())[i]
                            break
            return res
        except (MathDomainError, ZeroDivisionError):
            assert False, 'Math doing error\nCheck values ones more.'


if __name__ == '__main__':
    doctest = __import__('import doctest')
    doctest.testmod()