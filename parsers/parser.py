from typing import Sequence

import parsers.ast as ast
from parsers.lexer import Token, TokenType, Lexer
from parsers.token_stream import TokenStream, TokenSelector, UnexpectedToken


class SyntacticError(Exception):
    """Syntactic error."""

    def __init__(self, message: str, reason: UnexpectedToken) -> None:
        super().__init__(message)
        self.reason: UnexpectedToken = reason


class Parser:
    """
    Grammar:
        program = statement | statement ';' program
        statement = assign | expr
        assign = name '=' expr
        block = '{' program '}'
        expr = binary_expr
        binary_expr[prio=n] = binary_arg[prio=n] | binary_arg[prio=n] '<binary_operator[prio=n]>' binary_expr[prio=n]
        binary_arg[prio=n] = binary_expr[prio=n+1] | unary_expr
        unary_expr = ( expr ) | cond | number | bool | name | func_call | '<unary_operator>' unary_expr
        cond = 'if' '(' expr ')' block ( 'elif' '(' expr ')' block )* ('else' block )?
        func_call = name '(' func_args ')' | name '(' ')'
        func_args = expr | expr ',' func_args
    """

    # Binary operators from the lowest priority to the highest
    BINARY_OPERATORS = (
        (TokenType.OR,),
        (TokenType.AND,),
        (TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE, TokenType.EQ),
        (TokenType.PLUS, TokenType.MINUS),
        (TokenType.MUL, TokenType.DIV),
    )

    # Supported unary operators
    UNARY_OPERATORS = (TokenType.PLUS, TokenType.MINUS, TokenType.NOT)

    def program(self, tokens: TokenStream, end_program: TokenSelector) -> ast.Node:
        """
        program = statement ';' program | statement
        """
        try:
            statements = []
            start = tokens.current.pos
            if not tokens.match(end_program):
                statement = self.statement(tokens)
                statements.append(statement)

            while not tokens.match(end_program):
                tokens.take(TokenType.SEMICOLON)
                statement = self.statement(tokens)
                statements.append(statement)
            return ast.Block(statements, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse code block", reason=e)

    def block(self, tokens: TokenStream) -> ast.Node:
        """
        block = '{' program '}'
        """
        try:
            tokens.take(TokenType.OPEN_CB)
            body = self.program(tokens, end_program=TokenType.CLOSE_CB)
            tokens.take(TokenType.CLOSE_CB)
            return body
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse code block", reason=e)

    def statement(self, tokens: TokenStream) -> ast.Node:
        """
        statement = assign | expr
        """
        if tokens.match(TokenType.NAME, TokenType.ASSIGN):
            return self.assign(tokens)
        else:
            return self.expr(tokens)

    def assign(self, tokens: TokenStream) -> ast.Node:
        """
        assign = name '=' expr
        """
        try:
            start = tokens.current.pos
            name = tokens.take(TokenType.NAME)
            tokens.take(TokenType.ASSIGN)
            expr = self.expr(tokens)
            return ast.Assign(name, expr, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse assignment", reason=e)

    def expr(self, tokens: TokenStream) -> ast.Node:
        """
        expr = binary_expr
        """
        return self.binary_expr(tokens)

    def binary_expr(self, tokens: TokenStream, priority: int = 0) -> ast.Node:
        """
        binary_expr[prio=n] = binary_arg[prio=n] | binary_arg[prio=n] '<binary_operator[prio=n]>' binary_expr[prio=n]
        """
        start = tokens.current.pos
        expr = self.binary_arg(tokens, priority)
        while tokens.match(self.BINARY_OPERATORS[priority]):
            operator = tokens.take(self.BINARY_OPERATORS[priority])
            right_arg = self.binary_arg(tokens, priority)
            expr = ast.BinaryOperator(operator, expr, right_arg, pos=start)
        return expr

    def binary_arg(self, tokens: TokenStream, priority: int = 0) -> ast.Node:
        """
        binary_arg[prio=n] = binary_expr[prio=n+1] | unary_expr
        """
        if priority + 1 < len(self.BINARY_OPERATORS):
            return self.binary_expr(tokens, priority + 1)
        return self.unary_expr(tokens)

    def unary_expr(self, tokens: TokenStream) -> ast.Node:
        """
        unary_expr = ( expr ) | number | bool | func_call | name | if | '<unary_operator>' unary_expr
        """
        try:
            if tokens.match(TokenType.OPEN_BRACKET):
                tokens.take(TokenType.OPEN_BRACKET)
                expr = self.expr(tokens)
                tokens.take(TokenType.CLOSE_BRACKET)
                return expr
            elif tokens.match(TokenType.NUMBER):
                value = tokens.take(TokenType.NUMBER)
                return ast.NumberLiteral(value)
            elif tokens.match(TokenType.BOOL):
                value = tokens.take(TokenType.BOOL)
                return ast.BoolLiteral(value)
            elif tokens.match(TokenType.NAME, TokenType.OPEN_BRACKET):
                return self.func_call(tokens)
            elif tokens.match(TokenType.NAME):
                name = tokens.take(TokenType.NAME)
                return ast.VarName(name)
            elif tokens.match(TokenType.IF):
                return self.cond(tokens)
            elif tokens.match(self.UNARY_OPERATORS):
                operator = tokens.take(self.UNARY_OPERATORS)
                arg = self.unary_expr(tokens)
                return ast.UnaryOperator(operator, arg, pos=operator.pos)
            else:
                expected = (
                    TokenType.OPEN_BRACKET, TokenType.NUMBER, TokenType.BOOL,
                    TokenType.NAME, TokenType.IF, *self.UNARY_OPERATORS
                )
                raise UnexpectedToken(tokens.current, expected)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse unary expression", reason=e)

    def func_call(self, tokens: TokenStream) -> ast.Node:
        """
        func_call = name '(' func_args ')' | name '(' ')'
        """
        try:
            name = tokens.take(TokenType.NAME)
            tokens.take(TokenType.OPEN_BRACKET)
            args = self.func_args(tokens, args_end=TokenType.CLOSE_BRACKET)
            tokens.take(TokenType.CLOSE_BRACKET)
            return ast.FuncCall(name, args, pos=name.pos)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse function call", reason=e)

    def func_args(self, tokens: TokenStream, args_end: TokenSelector = TokenType.CLOSE_BRACKET) -> Sequence[ast.Node]:
        """
        func_args = expr | expr ',' func_args
        """
        try:
            args = []
            if not tokens.match(args_end):
                arg = self.expr(tokens)
                args.append(arg)
            while not tokens.match(args_end):
                tokens.take(TokenType.COMMA)
                arg = self.expr(tokens)
                args.append(arg)
            return args
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse function arguments", reason=e)

    def cond(self, tokens: TokenStream) -> ast.Node:
        """
        cond = 'if' '(' expr ')' block ( 'elif' '(' expr ')' block )* ('else' block )?
        """
        try:
            start = tokens.current.pos
            tokens.take(TokenType.IF, TokenType.OPEN_BRACKET)
            expr = self.expr(tokens)
            tokens.take(TokenType.CLOSE_BRACKET)
            body = self.block(tokens)

            cond_blocks = [ast.Cond.Block(expr, body)]
            else_block = None

            while tokens.match(TokenType.ELIF):
                tokens.take(TokenType.ELIF)
                expr = self.expr(tokens)
                tokens.take(TokenType.CLOSE_BRACKET)
                body = self.block(tokens)
                cond_blocks.append(ast.Cond.Block(expr, body))

            if tokens.match(TokenType.ELSE):
                tokens.take(TokenType.ELSE)
                else_block = self.block(tokens)

            return ast.Cond(cond_blocks, else_block, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse conditional expression", reason=e)

    def parse(self, tokens: Sequence[Token]) -> ast.Node:
        ignore = (TokenType.SPACE, TokenType.NEW_LINE)
        tokens = tuple(token for token in tokens if token.type not in ignore)
        return self.program(TokenStream(tokens), end_program=TokenType.END)


def main():
    lexer = Lexer()
    parser = Parser()

    text = input()
    while text.strip() != "exit":
        tokens = lexer.tokens(text)
        syntax_tree = parser.parse(tokens)
        print("OK")
        text = input()


if __name__ == '__main__':
    main()
