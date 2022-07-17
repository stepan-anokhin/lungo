import abc
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Optional, Dict

from esperanto.lexer import Position




class Context:
    """Execution context contains known variable names."""

    @dataclass
    class Caller:
        pos: Position
        context: "Context"

    def __init__(self, parent: Optional["Context"] = None, symbols: Optional[Dict[str, Any]] = None,
                 caller: Caller = None):
        self._symbols: Dict[str, Any] = symbols or {}
        self._parent = parent
        self._caller = caller

    def define(self, name: str, value: Any, pos: Position):
        """Define variable."""
        if name in self._symbols:
            raise NameRedefined(name, self, pos)
        self._symbols[name] = value

    def assign(self, name: str, value: Any, pos: Position):
        """Assign variable."""
        if name in self._symbols:
            self._symbols[name] = value
        elif self._parent is not None:
            try:
                self._parent.assign(name, value, pos)
            except NameNotFound:
                raise NameNotFound(name, self, pos)
        else:
            raise NameNotFound(name, self, pos)

    def get(self, name: str, pos: Position):
        if name in self._symbols:
            return self._symbols[name]
        elif self._parent is not None:
            try:
                return self._parent.get(name, pos)
            except NameNotFound:
                raise NameNotFound(name, self, pos)
        else:
            raise NameNotFound(name, self, pos)


class _Returned(Exception):
    """Raised by return statement."""

    def __init__(self, value: Any):
        self.value: Any = value


class ExecutionError(Exception):
    """Parent exception for the program execution errors."""

    def __init__(self, message: str, context: Context, pos: Position):
        super().__init__(message)
        self.context: Context = context
        self.pos: Position = pos


class NameNotFound(ExecutionError):
    """Name not found."""

    def __init__(self, name: str, context: Context, pos: Position):
        super().__init__(f"Name not found: {name}", context, pos)
        self.name = name


class NameRedefined(ExecutionError):
    """Attempt to redefine name."""

    def __init__(self, name: str, context: Context, pos: Position):
        super().__init__(f"Cannot redefine name: {name}", context, pos)
        self.name = name


class Program(abc.ABC):
    """Abstract executable code."""
    pos: Position

    @abc.abstractmethod
    def execute(self, context: Context) -> Any:
        """Execute code."""


class Operator(Program):
    """Binary operator expression."""

    def __init__(self, operator: Callable, args: Sequence[Program], pos: Position):
        self.operator: Callable = operator
        self.args: Sequence[Program] = tuple(args)
        self.pos = pos

    def execute(self, context: Context) -> Any:
        values = []
        for arg in self.args:
            value = arg.execute(context)
            values.append(value)
        return self.operator(*values)


class Value(Program):
    """Constant value."""

    def __init__(self, value, pos: Position) -> None:
        self.value = value
        self.pos = pos

    def execute(self, context: Context) -> Any:
        return self.value


class NameRef(Program):
    """Name reference."""

    def __init__(self, name: str, pos: Position):
        self.name: str = name
        self.pos = pos

    def execute(self, context: Context) -> Any:
        return context.get(self.name, self.pos)


class Assign(Program):
    """Assign variable."""

    def __init__(self, name: str, value: Program, pos: Position):
        self.name: str = name
        self.value: Program = value
        self.pos = pos

    def execute(self, context: Context) -> Any:
        value = self.value.execute(context)
        context.assign(self.name, value, self.pos)
        return value


class Define(Program):
    """Define variable."""

    def __init__(self, name: str, value: Program, pos: Position):
        self.name: str = name
        self.value: Program = value
        self.pos = pos

    def execute(self, context: Context) -> Any:
        value = self.value.execute(context)
        context.define(self.name, value, self.pos)


class Block(Program):
    """Sequence of statements."""

    def __init__(self, statements: Sequence[Program], pos: Position):
        self.statements: Sequence[Program] = statements
        self.pos = pos

    def execute(self, context: Context) -> Any:
        value = None
        nested = Context(parent=context)
        for statement in self.statements:
            value = statement.execute(nested)
        return value


class FuncExpr(Program):
    """Function expression."""

    def __init__(self, name: Optional[str], arg_names: Sequence[str], body: Program, pos: Position):
        self.name: Optional[str] = name
        self.arg_names: Sequence[str] = arg_names
        self.body: Program = body
        self.pos: Position = pos

    def execute(self, context: Context) -> Any:
        func = Function(self.name, self.arg_names, self.body, context)
        if self.name is not None:
            context.define(self.name, func, self.pos)
        return func


class Function:
    """Function object."""

    def __init__(self, name: Optional[str], arg_names: Sequence[str], body: Program, lexical_context: Context):
        self.name: Optional[str] = name
        self.arg_names: Sequence[str] = arg_names
        self.body: Program = body
        self.lexical_context: Context = lexical_context

    def call(self, args: Sequence[Any], caller: Context.Caller):
        if len(args) != len(self.arg_names):
            raise ExecutionError(f"Wrong number of arguments to call function {self.name}", caller.context, caller.pos)
        arguments = dict(zip(self.arg_names, args))
        context = Context(parent=self.lexical_context, symbols=arguments, caller=caller)

        try:
            return self.body.execute(context)
        except _Returned as returned:
            return returned.value


class FuncCall(Program):
    """Function invocation."""

    def __init__(self, func: Program, args: Sequence[Program], pos: Position):
        self.func: Program = func
        self.args: Sequence[Program] = args
        self.pos = pos

    def execute(self, context: Context) -> Any:
        func = self.func.execute(context)
        if not isinstance(func, Function):
            raise ExecutionError(f"{type(func)} is not a function", context, self.pos)
        args = []
        for arg in self.args:
            arg_value = arg.execute(context)
            args.append(arg_value)
        return func.call(args, Context.Caller(self.pos, context))


class ListExpr(Program):
    """List literal expression."""

    def __init__(self, items: Sequence[Program], pos: Position):
        self.items: Sequence[Program] = items
        self.pos = pos

    def execute(self, context: Context) -> Any:
        item_values = []
        for item in self.items:
            value = item.execute(context)
            item_values.append(value)
        return item_values


class GetItem(Program):
    """Get item expression."""

    def __init__(self, coll: Program, index: Program, pos: Position):
        self.coll: Program = coll
        self.index: Program = index
        self.pos = pos

    def execute(self, context: Context) -> Any:
        coll = self.coll.execute(context)
        if not isinstance(coll, list):
            raise ExecutionError(f"{type(coll)} is not a list", context, self.pos)
        index = self.index.execute(context)
        return coll[index]


class Condition(Program):
    """Conditional expression."""

    @dataclass
    class Branch:
        cond: Program
        body: Program

    def __init__(self, branches: Sequence[Branch], else_block: Optional[Program], pos: Position):
        self.branches: Sequence[Condition.Branch] = branches
        self.else_block: Optional[Program] = else_block
        self.pos = pos

    def execute(self, context: Context) -> Any:
        for branch in self.branches:
            cond = branch.cond.execute(context)
            if cond:
                return branch.body.execute(context)
        if self.else_block is not None:
            return self.else_block.execute(context)


class Return(Program):
    """Return statement."""

    def __init__(self, value: Program, pos: Position):
        self.value: Program = value
        self.pos = pos

    def execute(self, context: Context) -> Any:
        value = self.value.execute(context)
        raise _Returned(value)
