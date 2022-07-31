from typing import Collection, Union, Sequence, Iterator

from lungo.lexer import Token, TokenType

TokenSelector = Union[TokenType, Collection[TokenType]]


class UnexpectedToken(Exception):
    def __init__(self, actual: Token, expected: TokenSelector) -> None:
        super().__init__(f"Unexpected token {actual}, expected {expected}")
        self.actual: Token = actual
        self.expected: TokenSelector = expected


def _match(token: Token, sel: TokenSelector) -> bool:
    """Match token with the selector."""
    if isinstance(sel, TokenType):
        return token.type == sel
    elif isinstance(sel, Collection):
        return token.type in sel
    else:
        raise ValueError(f"Unsupported token selector: {sel}")


def _unpack(items: Collection[Token]) -> Union[Collection[Token], Token]:
    """Replace the collection with single item by its only item."""
    if isinstance(items, Collection) and len(items) == 1:
        return next(iter(items))
    return items


class TokenStream:
    def __init__(self, tokens: Sequence[Token], position: int = 0):
        self.tokens: Sequence[Token] = tuple(tokens)
        self.position = position
        if len(self.tokens) == 0 or self.tokens[-1].type != TokenType.END:
            raise ValueError("The last token must be the token stream 'END'")

    def match(self, *selectors: TokenSelector) -> bool:
        """Check if tokens in the stream match the given selectors."""
        matched = 0
        for token, sel in zip(self, selectors):
            if not _match(token, sel):
                return False
            matched += 1
        return matched == len(selectors)

    def take(self, *selectors: TokenSelector) -> Union[Token, Sequence[Token]]:
        """Consume tokens from the stream while ensuring they match given selectors."""
        # Otherwise, take tokens if selectors are matched
        tokens = []
        for token, sel in zip(self, selectors):
            if not _match(token, sel):
                raise UnexpectedToken(token, sel)
            tokens.append(token)
        self.position = min(self.position + len(tokens), len(self.tokens) - 1)
        return _unpack(tokens)

    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens starting from the current position."""
        for i in range(self.position, len(self.tokens)):
            yield self.tokens[i]

    @property
    def current(self) -> Token:
        """Get current token."""
        return self.tokens[self.position]
