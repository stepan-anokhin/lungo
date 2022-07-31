from typing import Callable, Optional, Dict

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer as BasePromptLexer

from lungo.lexer import Lexer, TokenType

TokeStyles = Dict[TokenType, str]


class PromptLexer(BasePromptLexer):
    DEFAULT_STYLE: TokeStyles = {
        # Special
        TokenType.SPACE: "class:pygments.whitespace",
        TokenType.NEW_LINE: "class:pygments.whitespace",
        TokenType.END: "class:pygments.whitespace",
        TokenType.UNKNOWN: "class:pygments.text",

        # Keywords:
        TokenType.FOR: "class:pygments.keyword",
        TokenType.IN: "class:pygments.keyword",
        TokenType.WHILE: "class:pygments.keyword",
        TokenType.IF: "class:pygments.keyword",
        TokenType.ELIF: "class:pygments.keyword",
        TokenType.ELSE: "class:pygments.keyword",
        TokenType.FUNC: "class:pygments.keyword",
        TokenType.RETURN: "class:pygments.keyword",
        TokenType.LET: "class:pygments.keyword",

        # Punctuation:
        TokenType.OPEN_BRACKET: "class:pygments.punctuation",
        TokenType.CLOSE_BRACKET: "class:pygments.punctuation",
        TokenType.OPEN_SB: "class:pygments.punctuation",
        TokenType.CLOSE_SB: "class:pygments.punctuation",
        TokenType.OPEN_CB: "class:pygments.punctuation",
        TokenType.CLOSE_CB: "class:pygments.punctuation",
        TokenType.COMMA: "class:pygments.punctuation",
        TokenType.SEMICOLON: "class:pygments.punctuation",

        # Operators:
        TokenType.PLUS: "class:pygments.operator",
        TokenType.MINUS: "class:pygments.operator",
        TokenType.MUL: "class:pygments.operator",
        TokenType.DIV: "class:pygments.operator",
        TokenType.AND: "class:pygments.operator",
        TokenType.OR: "class:pygments.operator",
        TokenType.NOT: "class:pygments.operator",
        TokenType.DOT: "class:pygments.operator",
        TokenType.ASSIGN: "class:pygments.operator",
        TokenType.NE: "class:pygments.operator",
        TokenType.GT: "class:pygments.operator",
        TokenType.GE: "class:pygments.operator",
        TokenType.LE: "class:pygments.operator",
        TokenType.LT: "class:pygments.operator",
        TokenType.EQ: "class:pygments.operator",

        # Atoms:
        TokenType.NAME: "class:pygments.name",
        TokenType.NUMBER: "class:pygments.number",
        TokenType.STRING: "class:pygments.string",
        TokenType.BOOL: "class:pygments.literal",
    }

    def __init__(self, lexer: Lexer, style: Optional[TokeStyles] = None):
        self.lexer: Lexer = lexer
        self.token_styles: TokeStyles = style or self.DEFAULT_STYLE

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        tokens = self.lexer.tokens(text=document.text)

        lines: Dict[int, StyleAndTextTuples] = {}
        for token in tokens:
            line = lines.setdefault(token.pos.line, [])
            style = self.token_styles[token.type]
            line.append((style, token.text))

        def get_tokens(line_number: int) -> StyleAndTextTuples:
            """Get tokens by line."""

            return lines.get(line_number, [])

        return get_tokens


Prompt = Callable[[str], str]
