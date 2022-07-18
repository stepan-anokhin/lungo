import abc
from dataclasses import dataclass
from typing import Optional, Dict, List, Sequence, Union, Callable

from esperanto.lexer import Position

SymbolTable = Dict[str, "Value"]


class CallStack:
    """Represents function call stack."""

    @dataclass(frozen=True)
    class Frame:
        pos: Position

    def __init__(self, pos: Optional[Position] = None):
        self._frames: List[CallStack.Frame] = []
        if pos is not None:
            self._frames.append(CallStack.Frame(pos))

    def push(self, frame: Frame):
        self._frames.append(frame)

    def pop(self) -> Frame:
        return self._frames.pop()


class ScopeError(Exception):
    """Parent class for Scope errors."""


class UndefinedName(ScopeError):
    """Name is not defined in the given scope."""

    def __init__(self, name: str):
        super().__init__(f"Name is not defined in the scope: {name}")
        self.name: str = name


class NameRedefined(ScopeError):
    """Attempt to define name already existing in the scope."""

    def __init__(self, name: str):
        super().__init__(f"Cannot redefine variable name: {name}")
        self.name: str = name


class Scope:
    """Hierarchy of lexical scopes, a table of known variable names."""

    def __init__(self, symbols: Optional[SymbolTable] = None, parent: Optional["Scope"] = None):
        symbols = symbols or {}
        self._symbols: SymbolTable = {**symbols}
        self.parent: Optional[Scope] = parent

    def assign(self, name: str, value):
        if name in self._symbols:
            self._symbols[name] = value
        elif self.parent is not None:
            self.parent.assign(name, value)
        else:
            raise UndefinedName(name)

    def define(self, name: str, value):
        if name in self._symbols:
            raise NameRedefined(name)
        self._symbols[name] = value

    def deref(self, name: str):
        if name in self._symbols:
            return self._symbols[name]
        elif self.parent is not None:
            return self.parent.deref(name)
        else:
            raise UndefinedName(name)

    def nested(self, symbols: Optional[SymbolTable] = None) -> "Scope":
        return Scope(symbols=symbols, parent=self)


class ExecutionContext:
    def __init__(self, scope: Scope, stack: CallStack):
        self.scope: Scope = scope
        self.stack: CallStack = stack

    def nested(self, symbols: Optional[SymbolTable] = None) -> "ExecutionContext":
        """Create execution context with nested lexical scope."""
        return ExecutionContext(self.scope.nested(symbols), self.stack)

    def push(self, pos: Position, scope: Scope):
        """Push another stack frame and prepare function scope."""
        self.stack.push(CallStack.Frame(pos))
        return ExecutionContext(scope, self.stack)


class ExecutionError(Exception):
    def __init__(self, message, stack: CallStack, pos: Position):
        super().__init__(message)
        self.stack = stack
        self.pos = pos


class Value:
    """Parent class for all values."""
    type: "Type"

    def plus(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Plus operator."""
        raise ExecutionError(f"Can't add to {self.type.name}", context.stack, pos)

    def minus(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Minus operator."""
        raise ExecutionError(f"Can't subtract from {self.type.name}", context.stack, pos)

    def mul(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Multiplication operator."""
        raise ExecutionError(f"Can't multiply {self.type.name}", context.stack, pos)

    def div(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Division operator."""
        raise ExecutionError(f"Can't divide {self.type.name}", context.stack, pos)

    def le(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Lesser-equal operator."""
        raise ExecutionError(f"Can't check {self.type.name} is lesser than or equal to other value", context.stack, pos)

    def lt(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Lesser-than operator."""
        raise ExecutionError(f"Can't check if {self.type.name} is lesser than other value", context.stack, pos)

    def ge(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Greater-equal operator."""
        raise ExecutionError(f"Can't check if {self.type.name} greater or equal to other value", context.stack, pos)

    def gt(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Greater-than operator."""
        raise ExecutionError(f"Can't check if {self.type.name} is greater than other value", context.stack, pos)

    def eq(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Equality operator."""
        return Bool.instance(self is other)

    def and_(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Logical-and operator."""
        if self.is_truthy():
            return other
        return self

    def or_(self, other: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Logical-or operator."""
        if self.is_truthy():
            return self
        return other

    def not_(self, context: ExecutionContext, pos: Position) -> "Value":
        """Logical-not unary operator."""
        return Bool.instance(not self.is_truthy())

    def neg(self, context: ExecutionContext, pos: Position) -> "Value":
        """Negation unary operator."""
        raise ExecutionError(f"Can't negate {self.type.name}", context.stack, pos)

    def getitem(self, key: "Value", context: ExecutionContext, pos: Position) -> "Value":
        """Get collection element by index."""
        raise ExecutionError(f"Can't get item from {self.type.name}", context.stack, pos)

    def call(self, arguments: Sequence["Value"], context: ExecutionContext, pos: Position) -> "Value":
        """Invoke-function operator."""
        raise ExecutionError(f"{self.type.name} is not callable", context.stack, pos)

    def getattr(self, name: str, context: ExecutionContext, pos: Position) -> "Value":
        """Get-attribute operator."""
        if name == "type":
            return self.type
        raise ExecutionError(f"{self.type.name} doesn't have attribute '{name}'", context.stack, pos)

    def to_bool(self, context: ExecutionContext, pos: Position) -> "Bool":
        """Converto to boolean."""
        return Bool.instance(self.is_truthy())

    def to_num(self, context: ExecutionContext, pos: Position) -> "Number":
        """Convert to number."""
        raise ExecutionError(f"Can't convert {self.type.name} to {NumberType.name}", context.stack, pos)

    def to_str(self, context: ExecutionContext, pos: Position) -> "String":
        """Convert to string."""
        raise ExecutionError(f"Can't convert {self.type.name} to {StringType.name}", context.stack, pos)

    @staticmethod
    def is_truthy() -> bool:
        """Check if the value is truthy."""
        return True


class Operators:
    """Supported operators as static functions."""

    @staticmethod
    def plus(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.plus(b, context, pos)

    @staticmethod
    def minus(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.minus(b, context, pos)

    @staticmethod
    def mul(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.mul(b, context, pos)

    @staticmethod
    def div(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.div(b, context, pos)

    @staticmethod
    def lt(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.lt(b, context, pos)

    @staticmethod
    def le(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.le(b, context, pos)

    @staticmethod
    def gt(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.gt(b, context, pos)

    @staticmethod
    def ge(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.ge(b, context, pos)

    @staticmethod
    def eq(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.eq(b, context, pos)

    @staticmethod
    def and_(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.and_(b, context, pos)

    @staticmethod
    def or_(context: ExecutionContext, pos: Position, a: Value, b: Value) -> Value:
        return a.or_(b, context, pos)

    @staticmethod
    def neg(context: ExecutionContext, pos: Position, a: Value) -> Value:
        return a.neg(context, pos)

    @staticmethod
    def not_(context: ExecutionContext, pos: Position, a: Value) -> Value:
        return a.not_(context, pos)


class Type(Value):
    """Value type."""
    _instance: "Type"
    name: str = "Type"

    def __init__(self):
        self.type = self

    @classmethod
    def instance(cls) -> "Type":
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def getattr(self, name: str, context: ExecutionContext, pos: Position) -> "Value":
        if name == "name":
            return String(self.name)
        super().getattr(name, context, pos)

    def __repr__(self):
        return self.name


class BoolType(Type):
    """Type for boolean values."""
    name: str = "Bool"


class Bool(Value):
    """Boolean value."""
    type = BoolType.instance()

    _TRUE: "Bool"
    _FALSE: "Bool"

    def __init__(self, value: bool):
        self.value: bool = value

    @staticmethod
    def instance(value: bool) -> "Bool":
        if value:
            if not hasattr(Bool, "_TRUE") or Bool._TRUE is None:
                Bool._TRUE = Bool(True)
            return Bool._TRUE
        if not hasattr(Bool, "_FALSE") or Bool._FALSE is None:
            Bool._FALSE = Bool(False)
        return Bool._FALSE

    def eq(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Equality operator."""
        if isinstance(other, Bool):
            return Bool.instance(self.value == other.value)
        return Bool.instance(False)

    def to_bool(self, context: ExecutionContext, pos: Position) -> "Bool":
        """Converto to boolean."""
        return self

    def to_num(self, context: ExecutionContext, pos: Position) -> "Number":
        """Convert to number."""
        if self.value:
            return Number(1)
        return Number(0)

    def to_str(self, context: ExecutionContext, pos: Position) -> "String":
        return String(str(self))

    def is_truthy(self) -> bool:
        return self.value

    def __str__(self):
        if self.value:
            return "true"
        return "false"

    def __repr__(self):
        return f"{self.type.name}({str(self)})"


class NilType(Type):
    """Type for 'nil' value."""
    name: str = 'Nil'


class Nil(Value):
    """Nil value."""
    type = NilType
    instance: "Nil"

    def to_str(self, context: ExecutionContext, pos: Position) -> "String":
        """Convert to string."""
        return String(str(self))

    @staticmethod
    def is_truthy() -> bool:
        """Check if the value is truthy."""
        return False

    def __repr__(self):
        return "nil"


Nil.instance = Nil()


class NumberType(Type):
    """Type for number values."""
    name: str = "Number"


class Number(Value):
    """Number value."""
    type = NumberType.instance()

    def __init__(self, value: Union[int, float]):
        self.value: Union[int, float] = value

    def plus(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Plus operator."""
        other_value = other.to_num(context, pos).value
        return Number(self.value + other_value)

    def minus(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Minus operator."""
        other_value = other.to_num(context, pos).value
        return Number(self.value - other_value)

    def mul(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Multiplication operator."""
        other_value = other.to_num(context, pos).value
        return Number(self.value * other_value)

    def div(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Division operator."""
        other_value = other.to_num(context, pos).value
        if other_value == 0:
            raise ExecutionError("Cannot divide by zero", context.stack, pos)
        return Number(self.value / other_value)

    def le(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Lesser-equal operator."""
        other_value = other.to_num(context, pos).value
        return Bool.instance(self.value <= other_value)

    def lt(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Lesser-than operator."""
        other_value = other.to_num(context, pos).value
        return Bool.instance(self.value < other_value)

    def ge(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Greater-equal operator."""
        other_value = other.to_num(context, pos).value
        return Bool.instance(self.value >= other_value)

    def gt(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Greater-than operator."""
        other_value = other.to_num(context, pos).value
        return Bool.instance(self.value > other_value)

    def eq(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Equality operator."""
        if isinstance(other, Number):
            return Bool.instance(self.value == other.value)
        return Bool.instance(False)

    def neg(self, context: ExecutionContext, pos: Position) -> Value:
        """Negation unary operator."""
        return Number(-self.value)

    def to_num(self, context: ExecutionContext, pos: Position) -> "Number":
        """Convert to number."""
        return self

    def to_str(self, context: ExecutionContext, pos: Position) -> "String":
        """Convert to string."""
        return String(str(self))

    def __repr__(self):
        return f"{NumberType.name}({self.value})"

    def __str__(self):
        return str(self.value)


class StringType(Type):
    """Type for string values."""
    name: str = "String"


class String(Value):
    """String value."""
    type: Type = StringType.instance()

    def __init__(self, value: str):
        self.value: str = value

    def plus(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Plus operator."""
        if isinstance(other, String):
            return String(self.value + other.value)
        raise ExecutionError(f"Can't concatenate {StringType.name} with {other.type.name}", context.stack, pos)

    def mul(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Multiplication operator."""
        if isinstance(other, Number):
            return String(self.value * other.value)
        raise ExecutionError(f"Can't multiply {StringType.name} and {other.type.name}", context.stack, pos)

    def le(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Lesser-equal operator."""
        if isinstance(other, String):
            return Bool.instance(self.value <= other.value)
        raise ExecutionError(f"Can't compare {StringType.name} and {other.type.name}", context.stack, pos)

    def lt(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Lesser-than operator."""
        if isinstance(other, String):
            return Bool.instance(self.value < other.value)
        raise ExecutionError(f"Can't compare {StringType.name} and {other.type.name}", context.stack, pos)

    def ge(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Greater-equal operator."""
        if isinstance(other, String):
            return Bool.instance(self.value >= other.value)
        raise ExecutionError(f"Can't compare {StringType.name} and {other.type.name}", context.stack, pos)

    def gt(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Greater-than operator."""
        if isinstance(other, String):
            return Bool.instance(self.value > other.value)
        raise ExecutionError(f"Can't compare {StringType.name} and {other.type.name}", context.stack, pos)

    def eq(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Equality operator."""
        if isinstance(other, String):
            return Bool.instance(self.value == other.value)
        return Bool.instance(False)

    def getitem(self, key: Value, context: ExecutionContext, pos: Position) -> Value:
        """Get collection element by index."""
        if isinstance(key, Number):
            return String(self.value[int(key.value)])
        raise ExecutionError(f"{String.type.name} indices must be {NumberType.name}", context.stack, pos)

    def getattr(self, name: str, context: ExecutionContext, pos: Position) -> Value:
        """Get-attribute operator."""
        if name == "length":
            return Number(len(self.value))
        super().getattr(name, context, pos)

    def to_str(self, context: ExecutionContext, pos: Position) -> "String":
        """Convert to string."""
        return self

    def is_truthy(self) -> bool:
        """Check if the value is truthy."""
        return len(self.value) > 0

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"{self.type.name}({repr(self.value)})"


class ListType(Type):
    """Type of list values."""
    name = "List"


class ListValue(Value):
    """List value."""
    type = ListType

    def __init__(self, items: List[Value]):
        self.items: List[Value] = items

    def plus(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Plus operator."""
        if isinstance(other, ListValue):
            return ListValue(self.items + other.items)
        raise ExecutionError(f"Can't add {other.type.name} to {ListType.name}", context.stack, pos)

    def mul(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Multiplication operator."""
        if isinstance(other, Number):
            return ListValue(self.items * int(other.value))
        raise ExecutionError(f"Can't multiply {ListType.name} and {other.type.name}", context.stack, pos)

    def le(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Lesser-equal operator."""
        if isinstance(other, ListValue):
            return Bool.instance(self.items <= other.items)
        raise ExecutionError(f"Can't compare {ListType.name} and {other.type.name}", context.stack, pos)

    def lt(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Lesser-than operator."""
        if isinstance(other, ListValue):
            return Bool.instance(self.items < other.items)
        raise ExecutionError(f"Can't compare {ListType.name} and {other.type.name}", context.stack, pos)

    def ge(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Greater-equal operator."""
        if isinstance(other, ListValue):
            return Bool.instance(self.items >= other.items)
        raise ExecutionError(f"Can't compare {ListType.name} and {other.type.name}", context.stack, pos)

    def gt(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Greater-than operator."""
        if isinstance(other, ListValue):
            return Bool.instance(self.items > other.items)
        raise ExecutionError(f"Can't compare {ListType.name} and {other.type.name}", context.stack, pos)

    def eq(self, other: Value, context: ExecutionContext, pos: Position) -> Value:
        """Equality operator."""
        if isinstance(other, ListValue):
            return Bool.instance(self.items == other.items)
        return Bool.instance(False)

    def getitem(self, key: Value, context: ExecutionContext, pos: Position) -> Value:
        """Get collection element by index."""
        if isinstance(key, Number):
            return self.items[int(key.value)]
        raise ExecutionError(f"{ListType.name} indices must be {NumberType.name}", context.stack, pos)

    def getattr(self, name: str, context: ExecutionContext, pos: Position) -> Value:
        """Get-attribute operator."""
        if name == "length":
            return Number(len(self.items))
        return super().getattr(name, context, pos)

    def to_str(self, context: ExecutionContext, pos: Position) -> String:
        """Convert to string."""
        items = ", ".join(item.to_str(context, pos).value for item in self.items)
        return String(f"[{items}]")

    def is_truthy(self) -> bool:
        """Check if the value is truthy."""
        return len(self.items) > 0

    def __str__(self):
        items = ", ".join(str(item) for item in self.items)
        return f"[{items}]"

    def __repr__(self):
        items = ", ".join(repr(item) for item in self.items)
        return f"[{items}]"


class FunctionType(Type):
    """Type of Function values."""
    name = "Function"


class _Returned(Exception):
    """Raised by return statement."""

    def __init__(self, value: Value):
        self.value: Value = value


class ExecutableCode(abc.ABC):
    """Abstract executable code."""
    pos: Position

    @abc.abstractmethod
    def execute(self, context: ExecutionContext) -> "Value":
        """Execute code."""


class Function(Value):
    """Function value."""

    def __init__(self, name: Optional[str], arg_names: Sequence[str], body: ExecutableCode, lexical_context: Scope):
        self.name: Optional[str] = name
        self.arg_names: Sequence[str] = arg_names
        self.body: ExecutableCode = body
        self.lexical_context: Scope = lexical_context

    def call(self, arguments: Sequence[Value], context: ExecutionContext, pos: Position) -> Value:
        """Invoke-function operator."""
        if len(arguments) != len(self.arg_names):
            raise ExecutionError(f"Wrong number of arguments to call {repr(self)}", context.stack, pos)
        arguments = dict(zip(self.arg_names, arguments))
        scope = self.lexical_context.nested(arguments)
        try:
            return self.body.execute(context.push(pos, scope))
        except _Returned as returned:
            return returned.value
        finally:
            context.stack.pop()

    def getattr(self, name: str, context: ExecutionContext, pos: Position) -> "Value":
        """Get-attribute operator."""
        if name == "name":
            return String(self.name or "")
        return super().getattr(name, context, pos)

    def to_str(self, context: ExecutionContext, pos: Position) -> String:
        """Convert to string."""
        return String(repr(self))

    def __repr__(self):
        name = self.name or "func"
        args = ', '.join(self.arg_names)
        return f"{name}({args})"


class Operator(ExecutableCode):
    """Binary operator expression."""

    def __init__(self, operator: Callable, args: Sequence[ExecutableCode], pos: Position):
        self.operator: Callable = operator
        self.args: Sequence[ExecutableCode] = tuple(args)
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        values: List[Value] = []
        for arg in self.args:
            value = arg.execute(context)
            values.append(value)
        return self.operator(context, self.pos, *values)


class Literal(ExecutableCode):
    """Constant value."""

    def __init__(self, value: Value, pos: Position) -> None:
        self.value: Value = value
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        return self.value


class NameRef(ExecutableCode):
    """Name reference."""

    def __init__(self, name: str, pos: Position):
        self.name: str = name
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        try:
            return context.scope.deref(self.name)
        except UndefinedName:
            raise ExecutionError(f"Undefined name '{self.name}'", context.stack, self.pos)


class Assign(ExecutableCode):
    """Assign variable."""

    def __init__(self, name: str, value: ExecutableCode, pos: Position):
        self.name: str = name
        self.value: ExecutableCode = value
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        value = self.value.execute(context)
        try:
            context.scope.assign(self.name, value)
            return value
        except UndefinedName:
            raise ExecutionError(f"Undefined name '{self.name}'", context.stack, self.pos)


class Define(ExecutableCode):
    """Define variable."""

    def __init__(self, name: str, value: ExecutableCode, pos: Position):
        self.name: str = name
        self.value: ExecutableCode = value
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        value = self.value.execute(context)
        try:
            context.scope.define(self.name, value)
            return value
        except NameRedefined:
            raise ExecutionError(f"Can't redefine variable '{self.name}'", context.stack, self.pos)


class Block(ExecutableCode):
    """Sequence of statements."""

    def __init__(self, statements: Sequence[ExecutableCode], pos: Position):
        self.statements: Sequence[ExecutableCode] = statements
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        value = None
        for statement in self.statements:
            value = statement.execute(context)
        return value


class FuncExpr(ExecutableCode):
    """Function expression."""

    def __init__(self, name: Optional[str], arg_names: Sequence[str], body: ExecutableCode, pos: Position):
        self.name: Optional[str] = name
        self.arg_names: Sequence[str] = arg_names
        self.body: ExecutableCode = body
        self.pos: Position = pos

    def execute(self, context: ExecutionContext) -> Value:
        func = Function(self.name, self.arg_names, self.body, context.scope)
        if self.name is not None:
            try:
                context.scope.define(self.name, func)
            except NameRedefined:
                raise ExecutionError(f"Can't redefine variable '{self.name}'", context.stack, self.pos)
        return func


class FuncCall(ExecutableCode):
    """Function invocation."""

    def __init__(self, func: ExecutableCode, args: Sequence[ExecutableCode], pos: Position):
        self.func: ExecutableCode = func
        self.args: Sequence[ExecutableCode] = args
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        func = self.func.execute(context)
        args = []
        for arg in self.args:
            arg_value = arg.execute(context)
            args.append(arg_value)
        return func.call(args, context, self.pos)


class ListExpr(ExecutableCode):
    """List literal expression."""

    def __init__(self, items: Sequence[ExecutableCode], pos: Position):
        self.items: Sequence[ExecutableCode] = items
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        item_values: List[Value] = []
        for item in self.items:
            value = item.execute(context)
            item_values.append(value)
        return ListValue(item_values)


class GetItem(ExecutableCode):
    """Get item expression."""

    def __init__(self, coll: ExecutableCode, index: ExecutableCode, pos: Position):
        self.coll: ExecutableCode = coll
        self.index: ExecutableCode = index
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        coll = self.coll.execute(context)
        index = self.index.execute(context)
        return coll.getitem(index, context, self.pos)


class GetAttr(ExecutableCode):
    """Get attribute by name."""

    def __init__(self, value: ExecutableCode, attr: str, pos: Position):
        self.value: ExecutableCode = value
        self.attr: str = attr
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        value = self.value.execute(context)
        return value.getattr(self.attr, context, self.pos)


class Condition(ExecutableCode):
    """Conditional expression."""

    @dataclass
    class Branch:
        cond: ExecutableCode
        body: ExecutableCode

    def __init__(self, branches: Sequence[Branch], else_block: Optional[ExecutableCode], pos: Position):
        self.branches: Sequence[Condition.Branch] = branches
        self.else_block: Optional[ExecutableCode] = else_block
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        for branch in self.branches:
            cond = branch.cond.execute(context)
            if cond.is_truthy():
                return branch.body.execute(context.nested())
        if self.else_block is not None:
            return self.else_block.execute(context.nested())


class While(ExecutableCode):
    """While loop."""

    def __init__(self, cond: ExecutableCode, body: ExecutableCode, pos: Position):
        self.cond: ExecutableCode = cond
        self.body: ExecutableCode = body
        self.pos = pos

    def execute(self, context: ExecutionContext) -> Value:
        value = Nil.instance
        while self.cond.execute(context).is_truthy():
            value = self.body.execute(context)
        return value


class Return(ExecutableCode):
    """Return statement."""

    def __init__(self, value: ExecutableCode, pos: Position):
        self.value: ExecutableCode = value
        self.pos = pos

    def execute(self, context: ExecutionContext):
        value = self.value.execute(context)
        raise _Returned(value)
