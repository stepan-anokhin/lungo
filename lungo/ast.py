from dataclasses import dataclass
from typing import Sequence, Optional

from lungo.lexer import Position, Token


class Node:
    """Basic node class."""
    pos: Position


class BinaryOperator(Node):
    def __init__(self, operator: Token, left: Node, right: Node, pos: Position):
        self.operator: Token = operator
        self.left: Node = left
        self.right: Node = right
        self.pos = pos


class UnaryOperator(Node):
    def __init__(self, operator: Token, arg: Node, pos: Position):
        self.operator: Token = operator
        self.arg: Node = arg
        self.pos = pos


class NumberLiteral(Node):
    def __init__(self, value: Token):
        self.value: Token = value
        self.pos = value.pos


class BoolLiteral(Node):
    def __init__(self, value: Token):
        self.value: Token = value
        self.pos = value.pos


class StringLiteral(Node):
    def __init__(self, value: Token):
        self.value: Token = value
        self.pos = value.pos


class List(Node):
    def __init__(self, items: Sequence[Node], pos: Position):
        self.items: Sequence[Node] = items
        self.pos = pos


class NameRef(Node):
    def __init__(self, name: Token):
        self.name: Token = name
        self.pos = name.pos


class FuncCall(Node):
    def __init__(self, func: Node, args: Sequence[Node], pos: Position):
        self.func: Node = func
        self.args: Sequence[Node] = tuple(args)
        self.pos = pos


class GetItem(Node):
    def __init__(self, list_expr: Node, index: Node, pos: Position):
        self.coll: Node = list_expr
        self.key: Node = index
        self.pos = pos


class Assign(Node):
    def __init__(self, name: Token, value: Node, pos: Position):
        self.name: Token = name
        self.value: Node = value
        self.pos = pos


class Block(Node):
    def __init__(self, statements: Sequence[Node], pos: Position):
        self.statements: Sequence[Node] = tuple(statements)
        self.pos = pos


class FuncExpr(Node):
    def __init__(self, name: Optional[Token], arg_names: Sequence[Token], body: Node, pos: Position):
        self.name: Optional[Token] = name
        self.arg_names: Sequence[Token] = arg_names
        self.body = body
        self.pos = pos


class DefineVar(Node):
    def __init__(self, name: Token, value: Node, pos: Position):
        self.name: Token = name
        self.value: Node = value
        self.pos = pos


class Return(Node):
    def __init__(self, value: Node, pos: Position):
        self.value: Node = value
        self.pos = pos


class Cond(Node):
    @dataclass
    class Block:
        cond: Node
        body: Node

    def __init__(self, cond_blocks: Sequence[Block], else_block: Optional[Node], pos: Position):
        self.cond_blocks = tuple(cond_blocks)
        self.else_block = else_block
        self.pos = pos


class While(Node):
    def __init__(self, cond: Node, body: Node, pos: Position):
        self.cond: Node = cond
        self.body: Node = body
        self.pos = pos


class For(Node):
    def __init__(self, name: Token, iterable: Node, body: Node, pos: Position):
        self.name: Token = name
        self.iterable: Node = iterable
        self.body: Node = body
        self.pos = pos


class GetAttr(Node):
    def __init__(self, value: Node, attr: Token, pos: Position):
        self.value: Node = value
        self.attr: Token = attr
        self.pos = pos


class SetAttr(Node):
    def __init__(self, target: Node, attr: Token, value: Node, pos: Position):
        self.target: Node = target
        self.value: Node = value
        self.attr: Token = attr
        self.pos = pos


class SetItem(Node):
    def __init__(self, coll: Node, key: Node, value: Node, pos: Position):
        self.coll: Node = coll
        self.key: Node = key
        self.value: Node = value
        self.pos = pos
