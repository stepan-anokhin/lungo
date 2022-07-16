import enum
import re
from dataclasses import dataclass
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


@dataclass
class Position:
    """Position in source code."""
    abs: int  # Absolute position in characters
    line: int  # Line number
    in_line: int  # Position in line

    def __repr__(self):
        return f"{self.abs}:{self.line}:{self.in_line}"


class Token:
    """Represents a token in a text."""

    def __init__(self, token_type: TokenType, text: str, pos: Position):
        self.type: TokenType = token_type
        self.text: str = text
        self.pos: Position = pos

    def __repr__(self):
        return f"<{self.type.name} {repr(self.text)} at={self.pos}>"


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
        abs_pos, line_num, line_start = 0, 0, 0
        # Iterate over known patterns
        for match in self.regex.finditer(text):
            # Get position of the found pattern match
            start, end = match.span()

            # If we skipped some text between
            # previous match and current match,
            # mark the skipped text as UNKNOWN
            if start != abs_pos:
                skipped_pos = Position(abs_pos, line_num, abs_pos - line_start)
                yield Token(TokenType.UNKNOWN, text[abs_pos:start], skipped_pos)

            # Produce a new token
            token_type = TokenType[match.lastgroup]
            yield Token(token_type, text[start:end], Position(start, line_num, start - line_start))

            # Update position
            abs_pos = end
            if token_type == TokenType.NEW_LINE:
                line_num += 1
                line_start = end

        # Mark the remaining portion of the text as UNKNOWN
        # because it doesn't contain any known patterns
        if abs_pos != len(text):
            yield Token(TokenType.UNKNOWN, text[abs_pos:], Position(abs_pos, line_num, abs_pos - line_start))

        # Produce the END token when we processed entire text
        yield Token(TokenType.END, '', Position(len(text), line_num, len(text) - line_start))

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
