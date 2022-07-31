import sys
from typing import Optional

import lungo.runtime as rt
from lungo.lexer import Lexer
from lungo.parser import Parser, SyntacticError
from lungo.translator import Translator


class Interpreter:
    def __init__(self, lexer: Optional[Lexer] = None, parser: Optional[Parser] = None,
                 translator: Optional[Translator] = None):
        self._lexer: Lexer = lexer or Lexer()
        self._parser: Parser = parser or Parser()
        self._translator: Translator = translator or Translator()

    @staticmethod
    def make_global_scope(**custom_vars: rt.Value) -> rt.Scope:
        """Create global scope with standard library definitions."""
        return rt.Scope(symbols={
            rt.Type.name: rt.Type.instance(),
            rt.ListType.name: rt.ListType.instance(),
            rt.BoolType.name: rt.BoolType.instance(),
            rt.NumberType.name: rt.NumberType.instance(),
            rt.StringType.name: rt.StringType.instance(),
            rt.FunctionType.name: rt.FunctionType.instance(),
            rt.NilType.name: rt.NilType,
            "nil": rt.Nil.instance,
            **custom_vars
        })

    @staticmethod
    def make_context(scope: rt.Scope, code: rt.ExecutableCode) -> rt.ExecutionContext:
        """Create execution context."""
        return rt.ExecutionContext(scope, rt.CallStack(code.pos))

    def translate_sources(self, sources: str, filename: str) -> rt.ExecutableCode:
        """Convert sources to executable code."""
        tokens = self._lexer.tokens(sources, filename)
        syntax_tree = self._parser.parse(tokens)
        return self._translator.translate(syntax_tree)

    def repl(self):
        global_scope = self.make_global_scope(prompt=rt.String("lungo> "))
        command = input(global_scope.deref("prompt"))
        commands_count = 0
        while True:
            try:
                code = self.translate_sources(command, f"<command:{commands_count}>")
                context = self.make_context(global_scope, code)
                value = code.execute(context)
                print(repr(value))
            except SyntacticError as error:
                print(str(error), file=sys.stderr)
            except rt.ExecutionError as error:
                print(str(error), file=sys.stderr)
            finally:
                command = input(global_scope.deref("prompt"))
                commands_count += 1


def main():
    interpreter = Interpreter()
    interpreter.repl()


if __name__ == '__main__':
    main()
