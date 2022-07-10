import enum
import re
from typing import Dict


class TokenType(enum.Enum):
    """Token types."""
    SPACE = "SPACE"
    OPEN_BRACKET = "OPEN_BRACKET"
    CLOSE_BRACKET = "CLOSE_BRACKET"
    NUMBER = "NUMBER"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MUL = "MUL"
    DIV = "DIV"
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
        TokenType.SPACE: r"\s+",
        TokenType.NUMBER: r"\d+",
        TokenType.OPEN_BRACKET: r"\(",
        TokenType.CLOSE_BRACKET: r"\)",
        TokenType.PLUS: r"\+",
        TokenType.MINUS: r"\-",
        TokenType.MUL: r"\*",
        TokenType.DIV: r"/",
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


# lexer = Lexer()
# for token in lexer.tokens("x"):
#     print(token)
