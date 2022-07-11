import enum
import operator as std_operators
from typing import List, Tuple, Dict, Any, Sequence, Union

from parsers.lexer import Token, TokenType, Lexer


class SyntacticError(Exception):
    """Syntactic error."""

    def __init__(self, token: Token, expected_type: TokenType = None):
        super().__init__(f"Syntactic error at {token.start_position}")
        self.token = token
        self.expected_type = expected_type


class NodeTypes(enum.Enum):
    """Node type ids."""
    BINARY_OPERATOR = "BINARY_OPERATOR"
    UNARY_OPERATOR = "UNARY_OPERATOR"
    NUMBER_LITERAL = "NUMBER"
    VAR_NAME = "VAR_NAME"
    FUNC_CALL = "FUNC_CALL"
    ASSIGN = "ASSIGN"
    BLOCK = "BLOCK"


class Node:
    """Basic node class."""
    type: NodeTypes

    def execute(self, context):
        return None


class BinaryOperator(Node):
    supported_operators = {
        "+": std_operators.add,
        "-": std_operators.sub,
        "*": std_operators.mul,
        "/": std_operators.truediv
    }

    def __init__(self, operator: str, left: Node, right: Node):
        self.type: NodeTypes = NodeTypes.BINARY_OPERATOR
        self.operator: str = operator
        self.left: Node = left
        self.right: Node = right

    def execute(self, context):
        if self.operator in self.supported_operators:
            func = self.supported_operators[self.operator]
            return func(self.left.execute(context), self.right.execute(context))
        else:
            raise RuntimeError("Unsupported binary operator:", self.operator)


class UnaryOperator(Node):
    supported_operators = {
        "-": std_operators.neg,
        "+": lambda x: x,
    }

    def __init__(self, operator, arg):
        self.type: NodeTypes = NodeTypes.UNARY_OPERATOR
        self.operator: str = operator
        self.arg: Node = arg

    def execute(self, context):
        if self.operator in self.supported_operators:
            func = self.supported_operators[self.operator]
            return func(self.arg.execute(context))
        else:
            raise RuntimeError("Unsupported unary operator:", self.operator)


class NumberLiteral(Node):
    def __init__(self, value):
        self.type = NodeTypes.NUMBER_LITERAL
        self.value = value

    def execute(self, context):
        return self.value


class VarName(Node):
    type = NodeTypes.VAR_NAME

    def __init__(self, name: str):
        self.name: str = name

    def execute(self, context):
        return context[self.name]


class FuncCall(Node):
    type = NodeTypes.FUNC_CALL

    def __init__(self, name: str, args: List[Node]):
        self.name: str = name
        self.args: List[Node] = args

    def execute(self, context):
        args = [arg.execute(context) for arg in self.args]
        func = context[self.name]
        return func(*args)


class Assign(Node):
    type = NodeTypes.ASSIGN

    def __init__(self, name, value):
        self.name: str = name
        self.value: Node = value

    def execute(self, context):
        value = self.value.execute(context)
        context[self.name] = value
        return value


class Block(Node):
    type = NodeTypes.BLOCK

    def __init__(self, statements: List[Node]):
        self.statements: List[Node] = statements

    def execute(self, context):
        value = None
        for statement in self.statements:
            value = statement.execute(context)
        return value


class TokenStream:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens: Sequence[Token] = tuple(tokens)
        self.position: int = 0

    def match(self, *types):
        for offset, expected_type in enumerate(types):
            if offset + self.position >= len(self.tokens):
                return False
            actual_type = self.tokens[self.position + offset].type
            if isinstance(expected_type, TokenType) and actual_type != expected_type:
                return False
            elif isinstance(expected_type, (list, tuple, set)) and actual_type not in expected_type:
                return False
        return True

    def take(self, *types) -> Union[Token, List[Token]]:
        if len(types) > 0:
            tokens = []
            for offset, expected_type in enumerate(types):
                if self.position + offset >= len(self.tokens):
                    raise SyntacticError(self.tokens[-1], expected_type)
                actual_token = self.tokens[self.position + offset]
                if isinstance(expected_type, TokenType) and actual_token.type != expected_type:
                    raise SyntacticError(actual_token, expected_type)
                elif isinstance(expected_type, (list, tuple, set)) and actual_token.type not in expected_type:
                    raise SyntacticError(actual_token)
                tokens.append(actual_token)
                self.position += 1
            if len(types) > 1:
                return tokens
            return tokens[0]
        if self.position >= len(self.tokens):
            raise SyntacticError(self.tokens[-1])
        token = self.tokens[self.position]
        self.position += 1
        return token

    def skip(self, *ignore: TokenType):
        while self.match(ignore):
            self.take()


class Parser:
    """
    program = statement ( ';' | \n ) program
    statement = assign | expr
    assign = name '=' expr
    expr = sum
    sum = term | term + sum | term - sum
    term = factor | factor * term | factor / term
    factor = ( sum ) | number | name | func_call | - factor | + factor
    func_call = name ( func_args ) | name ( )
    func_args = sum | sum , func_args
    """

    def program(self, tokens: TokenStream, end_program: TokenType) -> Node:
        """
        program = statement ( ';' | \n ) program
        """
        statements = []
        tokens.skip(TokenType.NEW_LINE)
        while not tokens.match(end_program):
            statement = self.statement(tokens)
            statements.append(statement)
            tokens.skip(TokenType.NEW_LINE, TokenType.SEMICOLON)
        return Block(statements)

    def statement(self, tokens: TokenStream) -> Node:
        """
        statement = assign | expr
        """
        if tokens.match(TokenType.NAME, TokenType.ASSIGN):
            return self.assign(tokens)
        else:
            return self.expr(tokens)

    def assign(self, tokens: TokenStream) -> Node:
        """
        assign = name '=' expr
        """
        name = tokens.take(TokenType.NAME).text
        tokens.take(TokenType.ASSIGN)
        expr = self.expr(tokens)
        return Assign(name, expr)

    def expr(self, tokens: TokenStream) -> Node:
        return self.sum(tokens)

    def sum(self, tokens: TokenStream) -> Node:
        """
        sum = term | term + sum | term - sum
        """
        term = self.term(tokens)
        if tokens.match([TokenType.PLUS, TokenType.MINUS]):
            operator = tokens.take()
            right = self.sum(tokens)
            return BinaryOperator(operator.text, term, right)
        return term

    def term(self, tokens: TokenStream) -> Node:
        """
        term = factor | factor * term | factor / term
        """
        factor = self.factor(tokens)
        if tokens.match([TokenType.MUL, TokenType.DIV]):
            operator = tokens.take()
            right = self.term(tokens)
            return BinaryOperator(operator.text, factor, right)
        return factor

    def factor(self, tokens: TokenStream) -> Node:
        """
        factor = ( expr ) | number | name | func_call | - factor | + factor
        """
        if tokens.match(TokenType.OPEN_BRACKET):
            tokens.take(TokenType.OPEN_BRACKET)
            expr = self.expr(tokens)
            tokens.take(TokenType.CLOSE_BRACKET)
            return expr
        elif tokens.match(TokenType.NUMBER):
            value = int(tokens.take(TokenType.NUMBER).text)
            return NumberLiteral(value)
        elif tokens.match(TokenType.NAME, TokenType.OPEN_BRACKET):
            return self.func_call(tokens)
        elif tokens.match(TokenType.NAME):
            return VarName(tokens.take(TokenType.NAME).text)
        elif tokens.match(TokenType.MINUS):
            tokens.take(TokenType.MINUS)
            arg = self.factor(tokens)
            return UnaryOperator("-", arg)
        elif tokens.match(TokenType.PLUS):
            tokens.take(TokenType.PLUS)
            return self.factor(tokens)
        else:
            raise SyntacticError(tokens.take())

    def func_call(self, tokens: TokenStream) -> Node:
        name = tokens.take(TokenType.NAME).text
        tokens.take(TokenType.OPEN_BRACKET)
        args = self.func_args(tokens, args_end=TokenType.CLOSE_BRACKET)
        tokens.take(TokenType.CLOSE_BRACKET)
        return FuncCall(name, args)

    def func_args(self, tokens: TokenStream, args_end=TokenType.CLOSE_BRACKET) -> List[Node]:
        args = []
        if not tokens.match(args_end):
            arg = self.expr(tokens)
            args.append(arg)
        while not tokens.match(args_end):
            tokens.take(TokenType.COMMA)
            arg = self.expr(tokens)
            args.append(arg)
        return args

    @staticmethod
    def _consume(tokens: List[Token], position: int, expected_type: TokenType) -> int:
        _, position = Parser._take(tokens, position, expected_type)
        return position

    @staticmethod
    def _take(tokens: List[Token], position: int, expected_type: TokenType) -> Tuple[Token, int]:
        next_token = tokens[position]
        if next_token.type == expected_type:
            return next_token, position + 1
        raise SyntacticError(next_token, expected_type)

    def parse(self, tokens: List[Token]) -> Node:
        tokens = TokenStream([token for token in tokens if token.type not in (TokenType.SPACE,)])
        expr = self.program(tokens, end_program=TokenType.END)
        tokens.take(TokenType.END)
        return expr


class Interpreter:
    def execute(self, text: str, context: Dict[str, Any]):
        try:
            lexer = Lexer()
            parser = Parser()
            tokens = lexer.tokens(text)
            syntactic_tree = parser.parse(tokens)
            return syntactic_tree.execute(context)
        except SyntacticError as err:
            self._print_error(err, text)

    @staticmethod
    def _print_error(err: SyntacticError, text: str):
        print(f"Syntactic error near char {err.token.start_position}:")
        print("\t", text)
        print("\t", " " * err.token.start_position + "^" * max(len(err.token.text), 1))
        print("Unexpected token:", err.token.type.name, repr(err.token.text))
        if err.expected_type is not None:
            print("Expected:", err.expected_type.name)


if __name__ == '__main__':
    interpreter = Interpreter()

    context = {
        "x": 42,
        "y": 1,
        "print": print,
        "pow": std_operators.pow,
    }

    text = input()
    while text.strip() != "exit":
        print("Result:", interpreter.execute(text, context))
        text = input()
