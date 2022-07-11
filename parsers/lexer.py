import enum
import re
from typing import Dict


class TokenType(enum.Enum):
    """Token types."""
    # White spaces
    SPACE = "SPACE"
    NEW_LINE = "NEW_LINE"

    # Structural
    OPEN_BRACKET = "OPEN_BRACKET"
    CLOSE_BRACKET = "CLOSE_BRACKET"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"

    # Atomic expressions
    NUMBER = "NUMBER"
    NAME = "NAME"

    # Complex expressions
    PLUS = "PLUS"
    MINUS = "MINUS"
    MUL = "MUL"
    DIV = "DIV"
    ASSIGN = "ASSIGN"

    # Technical
    UNKNOWN = "UNKNOWN"
    END = "END"


class Token:
    """Represents a token in a text."""

    def __init__(self, token_type, text, start_position):
        self.type = token_type
        self.text = text
        self.start_position = start_position

    def __repr__(self):
        return f"<{self.type.name} {repr(self.text)} at={self.start_position}>"


class Lexer:
    """Lexical analyzer."""
    default_patterns = {
        TokenType.SPACE: r"[ \t]+",
        TokenType.NEW_LINE: r"\n",
        TokenType.NAME: r"[a-zA-Z_][a-zA-Z_\d]*",
        TokenType.NUMBER: r"\d+",
        TokenType.OPEN_BRACKET: r"\(",
        TokenType.CLOSE_BRACKET: r"\)",
        TokenType.COMMA: r",",
        TokenType.PLUS: r"\+",
        TokenType.MINUS: r"\-",
        TokenType.MUL: r"\*",
        TokenType.DIV: r"/",
        TokenType.ASSIGN: r"=(?=[^=])"
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
        position = 0
        for match in self.regex.finditer(text):
            token_type = TokenType[match.lastgroup]
            start, end = match.span()
            if start != position:
                yield Token(TokenType.UNKNOWN, text[position:start], position)
            position = end
            yield Token(token_type, text[start:end], start)
        if position != len(text):
            yield Token(TokenType.UNKNOWN, text[position:], position)
        yield Token(TokenType.END, '', len(text))

    def tokens(self, text):
        return list(self.iter_tokens(text))


def main():
    lexer = Lexer()
    text = "hello\n   \n\nworld"
    while text.strip() not in ["\\q", "exit"]:
        for token in lexer.iter_tokens(text):
            print(token)
        text = input()


if __name__ == '__main__':
    main()
