import enum
import operator as std_operators
from typing import List, Tuple

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


class Node:
    """Basic node class."""
    type: NodeTypes

    def execute(self):
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

    def execute(self):
        if self.operator in self.supported_operators:
            func = self.supported_operators[self.operator]
            return func(self.left.execute(), self.right.execute())
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

    def execute(self):
        if self.operator in self.supported_operators:
            func = self.supported_operators[self.operator]
            return func(self.arg.execute())
        else:
            raise RuntimeError("Unsupported unary operator:", self.operator)


class NumberLiteral(Node):
    def __init__(self, value):
        self.type = NodeTypes.NUMBER_LITERAL
        self.value = value

    def execute(self):
        return self.value


class Parser:
    """
    sum = term | term + sum | term - sum

    term = factor | factor * term | factor / term

    factor = ( sum ) | number | - factor | + factor
    """

    def sum(self, tokens: List[Token], position: int) -> Tuple[Node, int]:
        term, position = self.term(tokens, position)
        if tokens[position].type == TokenType.PLUS:
            right, position = self.sum(tokens, position + 1)
            return BinaryOperator("+", term, right), position
        elif tokens[position].type == TokenType.MINUS:
            right, position = self.sum(tokens, position + 1)
            return BinaryOperator("-", term, right), position
        return term, position

    def term(self, tokens: List[Token], position: int) -> Tuple[Node, int]:
        factor, position = self.factor(tokens, position)
        if tokens[position].type == TokenType.MUL:
            right, position = self.term(tokens, position + 1)
            return BinaryOperator("*", factor, right), position
        elif tokens[position].type == TokenType.DIV:
            right, position = self.term(tokens, position + 1)
            return BinaryOperator("/", factor, right), position
        return factor, position

    def factor(self, tokens: List[Token], position: int) -> Tuple[Node, int]:
        """
        factor = ( sum ) | number | - factor | + factor
        """
        start_token = tokens[position]
        if start_token.type == TokenType.OPEN_BRACKET:
            expr, position = self.sum(tokens, position + 1)
            position = self._consume(tokens, position, TokenType.CLOSE_BRACKET)
            return expr, position
        elif start_token.type == TokenType.NUMBER:
            return NumberLiteral(value=int(start_token.text)), position + 1
        elif start_token.type == TokenType.MINUS:
            arg, position = self.factor(tokens, position + 1)
            return UnaryOperator("-", arg), position
        elif start_token.type == TokenType.PLUS:
            return self.factor(tokens, position + 1)
        else:
            raise SyntacticError(start_token)

    @staticmethod
    def _consume(tokens: List[Token], position: int, expected_type: TokenType) -> int:
        next_token = tokens[position]
        if next_token.type == expected_type:
            return position + 1
        raise SyntacticError(next_token, expected_type)

    def parse(self, tokens: List[Token]) -> Node:
        tokens = [token for token in tokens if token.type != TokenType.SPACE]
        expr, position = self.sum(tokens, position=0)
        self._consume(tokens, position, TokenType.END)
        return expr


class Interpreter:
    def execute(self, text: str):
        try:
            lexer = Lexer()
            parser = Parser()
            tokens = lexer.tokens(text)
            syntactic_tree = parser.parse(tokens)
            return syntactic_tree.execute()
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

    text = input()
    while text.strip() != "exit":
        print("Result:", interpreter.execute(text))
        text = input()
