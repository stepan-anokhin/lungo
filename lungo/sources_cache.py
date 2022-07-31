from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from lungo.lexer import Position


class _History:
    @dataclass
    class Entry:
        key: str
        value: str
        prev: Optional[_History.Entry] = None
        next: Optional[_History.Entry] = None

        def remove(self):
            if self.prev is not None:
                self.prev.next = self.next
            if self.next is not None:
                self.next.prev = self.prev
            self.next = None
            self.prev = None

        def insert_after(self, entry: _History.Entry):
            self.next = None
            self.prev = entry
            if entry is not None:
                self.next = entry.next
                entry.next = self
                if self.next is not None:
                    self.next.prev = self

    def __init__(self):
        self._start: Optional[_History.Entry] = None
        self._end: Optional[_History.Entry] = None

    def append(self, entry: _History.Entry) -> _History.Entry:
        """Append new entry."""
        entry.insert_after(self._end)
        if self._start is None:
            self._start = entry
            self._end = entry
        return entry

    def remove(self, entry: Optional[_History.Entry]):
        """Move entry to the end of history."""
        if entry is None:
            return
        if self._start is entry:
            self._start = entry.next
        if self._end is entry:
            self._end = entry.prev
        entry.remove()

    def make_latest(self, entry: _History.Entry):
        self.remove(entry)
        self.append(entry)

    def remove_first(self) -> Optional[_History.Entry]:
        """Remove the least recent history entry."""
        first = self._start
        self.remove(first)
        return first


class SourceCache:
    def __init__(self, max_entries: int = 1000):
        self._max_entries: int = max_entries
        self._permanent: Dict[str, str] = {}
        self._cached: Dict[str, _History.Entry] = {}
        self._history: _History = _History()

    def remove(self, filename: str):
        """Remove filename from cache."""
        if filename in self._permanent:
            del self._permanent[filename]
        if filename in self._cached:
            entry = self._cached.pop(filename)
            self._history.remove(entry)

    def add(self, text: str, filename: str, can_evict: bool = True):
        """Add new sources to the cache."""
        self.remove(filename)
        if can_evict:
            entry = _History.Entry(filename, text)
            self._history.append(entry)
            self._cached[filename] = entry
            if len(self._cached) > self._max_entries:
                self.evict()
        else:
            self._permanent[filename] = text

    def get(self, filename: str) -> str:
        """Get file text."""
        if filename in self._permanent:
            return self._permanent[filename]
        elif filename in self._cached:
            entry = self._cached[filename]
            self._history.make_latest(entry)
            return entry.value
        with open(filename) as file:
            text = file.read()
            self.add(text=text, filename=filename, can_evict=True)
            return text

    def evict(self):
        """Evict the least recently used item."""
        entry = self._history.remove_first()
        if entry is not None:
            del self._cached[entry.key]

    def get_line(self, pos: Position) -> str:
        """Get line in the file."""
        text = self.get(pos.file)
        try:
            start = text.rindex("\n", 0, pos.abs) + 1
        except ValueError:
            start = 0
        try:
            end = text.index("\n", pos.abs) or len(text)
        except ValueError:
            end = len(text)
        return text[start:end]
