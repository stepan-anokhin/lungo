import sys
from typing import Optional, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import style_from_pygments_cls, Style
from pygments.styles.monokai import MonokaiStyle

import lungo.runtime as rt
from lungo.lexer import Lexer
from lungo.parser import Parser, SyntacticError
from lungo.prompt import PromptLexer, Prompt
from lungo.translator import Translator


def print_one(value: rt.Value):
    print(value)


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
            "print": rt.PyFunc(print_one, name="print"),
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

    def make_prompt(
            self,
            prompt_lexer: Optional[PromptLexer] = None,
            style: Optional[Style] = None,
            output: TextIO = sys.stdout,
            input: TextIO = sys.stdin,
    ) -> Prompt:
        """Make prompter function."""
        prompt_session = PromptSession(
            enable_history_search=True,
            lexer=prompt_lexer or PromptLexer(self._lexer),
            style=style or style_from_pygments_cls(MonokaiStyle),
            input=create_input(input),
            output=create_output(output),
        )
        return prompt_session.prompt

    def repl(self, output: TextIO = sys.stdout, err_output: TextIO = sys.stderr, input: TextIO = sys.stdin):
        global_scope = self.make_global_scope(prompt=rt.String("lungo> "))

        prompt = self.make_prompt(input=input, output=output)
        try:
            command_count = 0
            command = prompt(str(global_scope.deref("prompt")))
        except (EOFError, KeyboardInterrupt):
            return

        while True:
            try:
                code = self.translate_sources(command, f"<command:{command_count}>")
                context = self.make_context(global_scope, code)
                value = code.execute(context)
                print(repr(value), file=output)
            except SyntacticError as error:
                print(str(error), file=err_output)
            except rt.ExecutionError as error:
                print(str(error), file=err_output)

            try:
                command = prompt(str(global_scope.deref("prompt")))
                command_count += 1
            except (EOFError, KeyboardInterrupt):
                break


def main():
    interpreter = Interpreter()
    interpreter.repl()


if __name__ == '__main__':
    main()
