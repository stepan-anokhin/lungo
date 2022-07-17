import operator as std

import esperanto.ast as ast
import esperanto.runtime as rt
from esperanto.lexer import TokenType, Lexer
from esperanto.parser import Parser


class InterpreterError(Exception):
    """Indicates interpreter error."""


class Interpreter:
    BINARY_OPERATORS = {
        TokenType.PLUS: std.add,
        TokenType.MINUS: std.sub,
        TokenType.MUL: std.mul,
        TokenType.DIV: std.truediv,
        TokenType.LT: std.lt,
        TokenType.LE: std.le,
        TokenType.GT: std.gt,
        TokenType.GE: std.ge,
        TokenType.EQ: std.eq,
        TokenType.AND: std.and_,
        TokenType.OR: std.or_,
    }

    UNARY_OPERATORS = {
        TokenType.MINUS: std.neg,
        TokenType.NOT: std.not_,
    }

    def translate(self, node: ast.Node) -> rt.Program:
        if isinstance(node, ast.BoolLiteral):
            return rt.Value(bool(node.value.text == "true"), node.pos)
        elif isinstance(node, ast.NumberLiteral):
            return rt.Value(int(node.value.text), node.pos)
        elif isinstance(node, ast.UnaryOperator):
            arg = self.translate(node.arg)
            return rt.Operator(self.UNARY_OPERATORS[node.operator.type], args=(arg,), pos=node.pos)
        elif isinstance(node, ast.BinaryOperator):
            left = self.translate(node.left)
            right = self.translate(node.right)
            return rt.Operator(self.BINARY_OPERATORS[node.operator.type], args=(left, right), pos=node.pos)
        elif isinstance(node, ast.Cond):
            branches = []
            for cond_block in node.cond_blocks:
                cond = self.translate(cond_block.cond)
                body = self.translate(cond_block.body)
                branches.append(rt.Condition.Branch(cond, body))
            else_block = None
            if node.else_block is not None:
                else_block = self.translate(node.else_block)
            return rt.Condition(branches, else_block, node.pos)
        elif isinstance(node, ast.NameRef):
            return rt.NameRef(node.name.text, node.pos)
        elif isinstance(node, ast.FuncExpr):
            name = None
            if node.name is not None:
                name = node.name.text
            arg_names = [arg.text for arg in node.arg_names]
            body = self.translate(node.body)
            return rt.FuncExpr(name, arg_names, body, node.pos)
        elif isinstance(node, ast.FuncCall):
            func = self.translate(node.func)
            args = [self.translate(arg) for arg in node.args]
            return rt.FuncCall(func, args, node.pos)
        elif isinstance(node, ast.List):
            items = []
            for item in node.items:
                items.append(self.translate(item))
            return rt.ListExpr(items, node.pos)
        elif isinstance(node, ast.GetItem):
            list_expr = self.translate(node.list)
            index = self.translate(node.index)
            return rt.GetItem(list_expr, index, node.pos)
        elif isinstance(node, ast.Assign):
            value = self.translate(node.value)
            return rt.Assign(node.name.text, value, node.pos)
        elif isinstance(node, ast.DefineVar):
            value = self.translate(node.value)
            return rt.Define(node.name.text, value, node.pos)
        elif isinstance(node, ast.Return):
            value = self.translate(node.value)
            return rt.Return(value, node.pos)
        elif isinstance(node, ast.Block):
            statements = [self.translate(statement) for statement in node.statements]
            return rt.Block(statements, node.pos)
        else:
            raise InterpreterError(f"Unsupported syntax construct: {type(node)}")


def main():
    lexer = Lexer()
    parser = Parser()
    interpreter = Interpreter()

    text = input()
    while text.strip() != "exit":
        tokens = lexer.tokens(text)
        syntax_tree = parser.parse(tokens)
        program = interpreter.translate(syntax_tree)
        print(program.execute(rt.Context()))
        text = input()


if __name__ == '__main__':
    main()
