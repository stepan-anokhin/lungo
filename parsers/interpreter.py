import abc
from typing import Any, Callable, Sequence


class Context:
    """Execution context contains known variable names."""


class Program(abc.ABC):
    """Abstract executable code."""

    @abc.abstractmethod
    def execute(self, context: Context) -> Any:
        """Execute code."""


class Operator(Program):
    """Binary operator expression."""

    def __init__(self, operator: Callable, *args: Program):
        self.args: Sequence[Program] = args
        self.operator: Callable = operator

    def execute(self, context: Context) -> Any:
        values = []
        for arg in self.args:
            value = arg.execute(context)
            values.append(value)
        return self.operator(*values)


class Value(Program):
    """Constant value."""

    def __init__(self, value) -> None:
        self.value = value

    def execute(self, context: Context) -> Any:
        return self.value


class NameRef(Program):
    """Name reference."""

    def __init__(self, name: str):
        self.name: str = name

    def execute(self, context: Context) -> Any:
        return context[self.name]


class Assign(Program):
    """Assign variable."""

    def __init__(self, name: str, value: Program):
        self.name: str = name
        self.value: Program = value

    def execute(self, context: Context) -> Any:
        value = self.value.execute(context)
        context[self.name] = value

