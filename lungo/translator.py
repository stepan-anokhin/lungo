import lungo.ast as ast
import lungo.runtime as rt
from lungo.lexer import TokenType, Position


class InterpreterError(Exception):
    """Indicates interpreter error."""


class StandardOperators:
    """Supported operators as static functions."""

    @staticmethod
    def plus(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.plus(b, context, pos)

    @staticmethod
    def minus(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.minus(b, context, pos)

    @staticmethod
    def mul(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.mul(b, context, pos)

    @staticmethod
    def div(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.div(b, context, pos)

    @staticmethod
    def lt(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.lt(b, context, pos)

    @staticmethod
    def le(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.le(b, context, pos)

    @staticmethod
    def gt(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.gt(b, context, pos)

    @staticmethod
    def ge(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.ge(b, context, pos)

    @staticmethod
    def eq(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.eq(b, context, pos)

    @staticmethod
    def and_(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.and_(b, context, pos)

    @staticmethod
    def or_(context: rt.ExecutionContext, pos: Position, a: rt.Value, b: rt.Value) -> rt.Value:
        return a.or_(b, context, pos)

    @staticmethod
    def neg(context: rt.ExecutionContext, pos: Position, a: rt.Value) -> rt.Value:
        return a.neg(context, pos)

    @staticmethod
    def not_(context: rt.ExecutionContext, pos: Position, a: rt.Value) -> rt.Value:
        return a.not_(context, pos)


class Translator:
    BINARY_OPERATORS = {
        TokenType.PLUS: StandardOperators.plus,
        TokenType.MINUS: StandardOperators.minus,
        TokenType.MUL: StandardOperators.mul,
        TokenType.DIV: StandardOperators.div,
        TokenType.LT: StandardOperators.lt,
        TokenType.LE: StandardOperators.le,
        TokenType.GT: StandardOperators.gt,
        TokenType.GE: StandardOperators.ge,
        TokenType.EQ: StandardOperators.eq,
        TokenType.AND: StandardOperators.and_,
        TokenType.OR: StandardOperators.or_,
    }

    UNARY_OPERATORS = {
        TokenType.MINUS: StandardOperators.neg,
        TokenType.NOT: StandardOperators.not_,
    }

    def translate(self, node: ast.Node) -> rt.ExecutableCode:
        if isinstance(node, ast.BoolLiteral):
            return rt.Literal(rt.Bool.instance(node.value.text == "true"), node.pos)
        elif isinstance(node, ast.NumberLiteral):
            return rt.Literal(rt.Number(int(node.value.text)), node.pos)
        elif isinstance(node, ast.StringLiteral):
            text = self._unescape_string(node.value.text)
            return rt.Literal(rt.String(text), node.pos)
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
        elif isinstance(node, ast.While):
            cond = self.translate(node.cond)
            body = self.translate(node.body)
            return rt.While(cond, body, node.pos)
        elif isinstance(node, ast.For):
            iterable = self.translate(node.iterable)
            body = self.translate(node.body)
            return rt.For(node.name.text, iterable, body, node.pos)
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
            coll = self.translate(node.coll)
            key = self.translate(node.key)
            return rt.GetItem(coll, key, node.pos)
        elif isinstance(node, ast.SetItem):
            coll = self.translate(node.coll)
            key = self.translate(node.key)
            value = self.translate(node.value)
            return rt.SetItem(coll, key, value, node.pos)
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
        elif isinstance(node, ast.GetAttr):
            value = self.translate(node.value)
            return rt.GetAttr(value, node.attr.text, node.pos)
        elif isinstance(node, ast.SetAttr):
            target = self.translate(node.target)
            attr = node.attr.text
            value = self.translate(node.value)
            return rt.SetAttr(target, attr, value, node.pos)
        else:
            raise InterpreterError(f"Unsupported syntax construct: {type(node)}")

    @staticmethod
    def _unescape_string(value: str) -> str:
        """Unescape string literal."""
        return value[1:-1].encode('utf-8').decode("unicode_escape")
