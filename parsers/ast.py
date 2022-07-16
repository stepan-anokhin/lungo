from dataclasses import dataclass
from typing import Sequence, Optional

from parsers.lexer import Position, Token


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


class VarName(Node):
    def __init__(self, name: Token):
        self.name: Token = name
        self.pos = name.pos


class FuncCall(Node):
    def __init__(self, name: Token, args: Sequence[Node], pos: Position):
        self.name: Token = name
        self.args: Sequence[Node] = tuple(args)
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


class Cond(Node):
    @dataclass
    class Block:
        cond: Node
        body: Node

    def __init__(self, cond_blocks: Sequence[Block], else_block: Optional[Node], pos: Position):
        self.cond_blocks = tuple(cond_blocks)
        self.else_block = else_block
        self.pos = pos
