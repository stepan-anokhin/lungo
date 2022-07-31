from __future__ import annotations

import sys
from typing import Optional, TextIO, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import style_from_pygments_cls, Style
from pygments.styles.monokai import MonokaiStyle

import lungo.runtime as rt
from lungo.lexer import Lexer, Position
from lungo.parser import Parser, SyntacticError
from lungo.prompt import PromptLexer, Prompt
from lungo.sources_cache import SourceCache
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
    def make_context(scope: rt.Scope) -> rt.ExecutionContext:
        """Create execution context."""
        return rt.ExecutionContext(scope, rt.CallStack())

    def translate_sources(self, sources: str, filename: str) -> rt.ExecutableCode:
        """Convert sources to executable code."""
        tokens = self._lexer.tokens(sources, filename)
        syntax_tree = self._parser.parse(tokens)
        return self._translator.translate(syntax_tree)

    @staticmethod
    def print_stacktrace(error: rt.ExecutionError, output: TextIO = sys.stderr, sources: Optional[SourceCache] = None):
        """Print stack-trace."""
        print("Stack trace:", file=output)
        for frame in error.stack:
            Interpreter.print_stack_entry(frame.pos, sources=sources, output=output)
        Interpreter.print_stack_entry(error.pos, sources=sources, output=output)
        print(f"{type(error).__name__}: {str(error)}")

    @staticmethod
    def print_stack_entry(pos: Position, sources: Optional[SourceCache] = None, output: TextIO = sys.stderr):
        print(f"  In file '{pos.file}', line {pos.line}", file=output)
        if sources is not None:
            line = sources.get_line(pos)
            print(f"    {line.lstrip()}", file=output)

    @staticmethod
    def print_syntactic_error(error: SyntacticError, output: TextIO = sys.stderr,
                              sources: Optional[SourceCache] = None):
        """Print syntactic error."""
        token = error.reason.actual
        pos = token.pos
        print(f"In file '{pos.file}', line {pos.line}", file=output)
        if sources is not None:
            print("  " + sources.get_line(pos), file=output)
            print("  " + " " * pos.in_line + "^" * len(token.text), file=output)
        print(f"Syntactic Error: {str(error)}", file=output)

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

        sources = SourceCache()
        prompt = self.make_prompt(input=input, output=output)
        try:
            command_count = 0
            command = prompt(str(global_scope.deref("prompt")))
        except (EOFError, KeyboardInterrupt):
            return

        while True:
            try:
                command_key = f"<command:{command_count}>"
                sources.add(text=command, filename=command_key, can_evict=False)
                code = self.translate_sources(command, command_key)
                context = self.make_context(global_scope)
                value = code.execute(context)
                print(repr(value), file=output)
            except SyntacticError as error:
                self.print_syntactic_error(error, output=err_output, sources=sources)
            except rt.ExecutionError as error:
                self.print_stacktrace(error, output=err_output, sources=sources)

            try:
                command = prompt(str(global_scope.deref("prompt")))
                command_count += 1
            except (EOFError, KeyboardInterrupt):
                break

    def exec_file(self, path: str):
        """Execute file."""
        sources = SourceCache()
        try:
            text = sources.get(path)
            code = self.translate_sources(text, path)
            global_scope = self.make_global_scope()
            context = self.make_context(global_scope)
            code.execute(context)
        except SyntacticError as error:
            self.print_syntactic_error(error, sources=sources)
            sys.exit(-1)
        except rt.ExecutionError as error:
            self.print_stacktrace(error, sources=sources)
            sys.exit(-1)


def main(args: Optional[Sequence[str]] = None):
    args = args or sys.argv
    interpreter = Interpreter()
    if len(args) == 1:
        interpreter.repl()
    elif len(args) == 2:
        interpreter.exec_file(args[1])
    else:
        print("Usage:", file=sys.stderr)
        print("  lungo [FILE]", file=sys.stderr)


if __name__ == '__main__':
    main(["", "/home/stepan/PycharmProjects/parsers/main.lg"])
