"""Preserve Unicode text while neutralizing terminal C0/DEL/C1 controls."""
import json


def printable(value, *, preserve_newlines=False):
    def escape(char):
        code = ord(char)
        if (code < 32 and not (preserve_newlines and char in '\n\t')) or 127 <= code <= 159:
            return f'\\u{code:04x}'
        return char
    return ''.join(escape(c) for c in str(value))


def json_text(value, **kwargs):
    # JSON already escapes C0; C1 remains literal under ensure_ascii=False.
    return printable(json.dumps(value, **kwargs), preserve_newlines=True)
