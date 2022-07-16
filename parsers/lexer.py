import enum
import re
from typing import Dict


class TokenType(enum.Enum):
    """Token types."""
    # White spaces
    SPACE = "SPACE"
    NEW_LINE = "NEW_LINE"

    # Structural
    OPEN_CB = "OPEN_CB"
    CLOSE_CB = "CLOSE_CB"
    OPEN_BRACKET = "OPEN_BRACKET"
    CLOSE_BRACKET = "CLOSE_BRACKET"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    IF = "IF"
    ELIF = "ELIF"
    ELSE = "ELSE"

    # Atomic expressions
    NUMBER = "NUMBER"
    NAME = "NAME"
    BOOL = "BOOL"

    # Complex expressions
    PLUS = "PLUS"
    MINUS = "MINUS"
    MUL = "MUL"
    DIV = "DIV"
    ASSIGN = "ASSIGN"
    NOT = "NOT"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    EQ = "EQ"
    NE = "NE"
    AND = "AND"
    OR = "OR"

    # Technical
    UNKNOWN = "UNKNOWN"
    END = "END"


class Token:
    """Represents a token in a text."""

    def __init__(self, token_type: TokenType, text: str, abs_pos: int, line: int, line_pos: int):
        self.type: TokenType = token_type
        self.text: str = text
        self.abs_pos: int = abs_pos  # absolute start position
        self.line: int = line  # line number
        self.line_pos: int = line_pos  # start position in line

    def __repr__(self):
        return f"<{self.type.name} {repr(self.text)} at={self.abs_pos}:{self.line}:{self.line_pos}>"


class Lexer:
    """Lexical analyzer."""
    default_patterns = {
        TokenType.SPACE: r"[ \t]+",
        TokenType.NEW_LINE: r"\n",
        TokenType.SEMICOLON: r";",
        TokenType.NAME: r"(?!(?:true|false|if|elif|else)([^a-zA-Z_\d]|$))[a-zA-Z_][a-zA-Z_\d]*",
        TokenType.NUMBER: r"\d+",
        TokenType.OPEN_BRACKET: r"\(",
        TokenType.CLOSE_BRACKET: r"\)",
        TokenType.OPEN_CB: r"\{",
        TokenType.CLOSE_CB: r"\}",
        TokenType.COMMA: r",",
        TokenType.PLUS: r"\+",
        TokenType.MINUS: r"\-",
        TokenType.MUL: r"\*",
        TokenType.DIV: r"/",
        TokenType.ASSIGN: r"=(?=[^=])",
        TokenType.IF: r"if",
        TokenType.ELIF: r"elif",
        TokenType.ELSE: r"else",
        TokenType.BOOL: r"true|false",
        TokenType.NOT: r"!(?=[^=])",
        TokenType.LT: r"<(?=[^=])",
        TokenType.LE: r"<=",
        TokenType.GT: r">(?=[^=])",
        TokenType.GE: r">=",
        TokenType.EQ: r"==",
        TokenType.NE: r"!=",
        TokenType.AND: r"&&",
        TokenType.OR: r"\|\|",
    }

    def __init__(self, patterns=None):
        self.patterns: Dict[TokenType, str] = patterns or Lexer.default_patterns
        regex_entries = []
        for token_type, pattern in self.patterns.items():
            regex_entries.append(f"(?P<{token_type.name}>{pattern})")
        final_regex = "|".join(regex_entries)
        self.regex: re.Pattern = re.compile(final_regex)

    def iter_tokens(self, text: str):
        """Split text in tokens."""
        position, line_number, line_start = 0, 0, 0
        for match in self.regex.finditer(text):
            token_type = TokenType[match.lastgroup]
            start, end = match.span()
            if start != position:
                yield Token(TokenType.UNKNOWN, text[position:start], position, line_number, position - line_start)
            position = end
            yield Token(token_type, text[start:end], start, line_number, start - line_start)
            if token_type == TokenType.NEW_LINE:
                line_number += 1
                line_start = end
        if position != len(text):
            yield Token(TokenType.UNKNOWN, text[position:], position, line_number, position - line_start)
        yield Token(TokenType.END, '', len(text), line_number, len(text) - line_start)

    def tokens(self, text):
        return list(self.iter_tokens(text))


def main():
    lexer = Lexer()
    text = "hello\n   \n\nworld$$"
    while text.strip() not in ["\\q", "exit"]:
        for token in lexer.iter_tokens(text):
            print(token)
        text = input()


if __name__ == '__main__':
    main()
