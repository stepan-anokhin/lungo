from typing import Sequence, Optional

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
        statement = assign | define | ret | expr
        assign = name '=' expr
        define = 'let' name '=' expr
        ret = 'return' expr
        block = '{' program '}'
        expr = binary_expr
        binary_expr[prio=n] = binary_arg[prio=n] | binary_arg[prio=n] '<binary_operator[prio=n]>' binary_expr[prio=n]
        binary_arg[prio=n] = binary_expr[prio=n+1] | prefix_expr
        prefix_expr = '<unary_operator>' prefix_expr | postfix_expr
        postfix_expr = postfix_arg postfix_operation
        postfix_arg = ( expr ) | list | cond | number | bool | name
        list = '[' expr_list ']'
        postfix_operation = func_call | get_item
        cond = 'if' '(' expr ')' block ( 'elif' '(' expr ')' block )* ('else' block )?
        func = 'func' name ? '(' name_list ')' block
        func_call = '(' expr_list ')'
        get_item = '[' expr ']'
        expr_list = expr | expr ',' expr_list
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
        statement = assign | define | ret | expr
        """
        if tokens.match(TokenType.NAME, TokenType.ASSIGN):
            return self.assign(tokens)
        elif tokens.match(TokenType.LET):
            return self.define_var(tokens)
        elif tokens.match(TokenType.RETURN):
            return self.ret(tokens)
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

    def define_var(self, tokens: TokenStream) -> ast.Node:
        """
        define = 'let' name '=' expr
        """
        try:
            start = tokens.current.pos
            tokens.take(TokenType.LET)
            name = tokens.take(TokenType.NAME)
            tokens.take(TokenType.EQ)
            value = self.expr(tokens)
            return ast.DefineVar(name, value, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse variable definition", reason=e)

    def ret(self, tokens: TokenStream) -> ast.Node:
        """
        ret = 'return' expr
        """
        try:
            start = tokens.current.pos
            tokens.take(TokenType.RETURN)
            value = self.expr(tokens)
            return ast.Return(value, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse return statement", reason=e)

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
        binary_arg[prio=n] = binary_expr[prio=n+1] | prefix_expr
        """
        if priority + 1 < len(self.BINARY_OPERATORS):
            return self.binary_expr(tokens, priority + 1)
        return self.prefix_expr(tokens)

    def prefix_expr(self, tokens: TokenStream) -> ast.Node:
        """
        prefix_expr = '<unary_operator>' prefix_expr | postfix_expr
        """
        try:
            if tokens.match(self.UNARY_OPERATORS):
                operator = tokens.take(self.UNARY_OPERATORS)
                arg = self.prefix_expr(tokens)
                return ast.UnaryOperator(operator, arg, pos=operator.pos)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse prefix expr", reason=e)
        return self.postfix_expr(tokens)

    def postfix_expr(self, tokens: TokenStream) -> ast.Node:
        """
        postfix_expr = postfix_arg postfix_operation
        """
        arg = self.postfix_arg(tokens)
        return self.postfix_operation(tokens, arg)

    def postfix_operation(self, tokens: TokenStream, arg: ast.Node) -> ast.Node:
        """
        postfix_operation = func_call | get_item
        """
        if tokens.match(TokenType.OPEN_BRACKET):
            func_call = self.func_call(tokens, arg)
            return self.postfix_operation(tokens, func_call)
        elif tokens.match(TokenType.OPEN_SB):
            get_item = self.get_item(tokens, arg)
            return self.postfix_operation(tokens, get_item)
        else:
            return arg

    def postfix_arg(self, tokens: TokenStream) -> ast.Node:
        """
        postfix_arg = '(' expr ')' | list | cond | func | number | bool | name
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
            elif tokens.match(TokenType.NAME):
                name = tokens.take(TokenType.NAME)
                return ast.NameRef(name)
            elif tokens.match(TokenType.IF):
                return self.cond(tokens)
            elif tokens.match(TokenType.FUNC):
                return self.func(tokens)
            elif tokens.match(TokenType.OPEN_SB):
                return self.list(tokens)
            else:
                expected = (
                    TokenType.OPEN_BRACKET, TokenType.NUMBER,
                    TokenType.BOOL, TokenType.NAME, TokenType.IF
                )
                raise UnexpectedToken(tokens.current, expected)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse postfix arg", reason=e)

    def list(self, tokens: TokenStream) -> ast.Node:
        """
        list = '[' expr_list ']'
        """
        try:
            start = tokens.current.pos
            tokens.take(TokenType.OPEN_SB)
            items = self.expr_list(tokens, list_end=TokenType.CLOSE_SB)
            tokens.take(TokenType.CLOSE_SB)
            return ast.List(items, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse list", reason=e)

    def get_item(self, tokens: TokenStream, list_expr: ast.Node) -> ast.Node:
        """
        get_item = '[' expr ']'
        """
        try:
            tokens.take(TokenType.OPEN_SB)
            index_expr = self.expr(tokens)
            tokens.take(TokenType.CLOSE_SB)
            return ast.GetItem(list_expr, index_expr, pos=list_expr.pos)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse get-item", reason=e)

    def func_call(self, tokens: TokenStream, func: ast.Node) -> ast.Node:
        """
        func_call = '(' expr_list ')' | '(' ')'
        """
        try:
            tokens.take(TokenType.OPEN_BRACKET)
            args = self.expr_list(tokens, list_end=TokenType.CLOSE_BRACKET)
            tokens.take(TokenType.CLOSE_BRACKET)
            return ast.FuncCall(func, args, pos=func.pos)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse function call", reason=e)

    def expr_list(self, tokens: TokenStream, list_end: TokenSelector = TokenType.CLOSE_BRACKET) -> Sequence[ast.Node]:
        """
        expr_list = expr | expr ',' expr_list
        """
        try:
            args = []
            if not tokens.match(list_end):
                arg = self.expr(tokens)
                args.append(arg)
            while not tokens.match(list_end):
                tokens.take(TokenType.COMMA)
                arg = self.expr(tokens)
                args.append(arg)
            return args
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse expression list", reason=e)

    def func(self, tokens: TokenStream) -> ast.Node:
        """
        func = 'func' name? '(' name_list ')' block
        """
        try:
            start = tokens.current.pos
            tokens.take(TokenType.FUNC)
            name: Optional[Token] = None
            if tokens.match(TokenType.NAME):
                name = tokens.take(TokenType.NAME)
            tokens.take(TokenType.OPEN_BRACKET)
            arg_names = self.name_list(tokens, list_end=TokenType.CLOSE_BRACKET)
            tokens.take(TokenType.CLOSE_BRACKET)
            body = self.block(tokens)
            return ast.FuncExpr(name, arg_names, body, pos=start)
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse function expression", reason=e)

    @staticmethod
    def name_list(tokens: TokenStream, list_end: TokenSelector = TokenType.CLOSE_BRACKET) -> Sequence[Token]:
        """
        name_list = name | name ',' name_list
        """
        try:
            names = []
            if not tokens.match(list_end):
                name = tokens.take(TokenType.NAME)
                names.append(name)
            while not tokens.match(list_end):
                tokens.take(TokenType.COMMA)
                name = tokens.take(TokenType.NAME)
                names.append(name)
            return names
        except UnexpectedToken as e:
            raise SyntacticError("Cannot parse name list", reason=e)

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
