"""Four truthful states for every number on the pulse tile and the dashboard.

The measure block is the same JavaScript in ``pulse.html`` and ``dashboard.js``.
These tests execute that block against fixtures. A null instrument renders
``not measurable yet``. A dream with no row renders ``not run yet``. The only
bare ``0`` is an empty memories table, and it carries ``the store is empty``.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from html.parser import HTMLParser

from tests._bootstrap import ROOT


import math
from datetime import datetime, timezone


def js_string(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return str(value)


def js_add(left, right):
    if isinstance(left, str) or isinstance(right, str) or left is None or right is None:
        return js_string(left) + js_string(right)
    return left + right


def js_typeof(value):
    if value is None:
        return "object"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    return "object"


def js_index(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return obj[key]


def js_round(value):
    if isinstance(value, bool) or value is None:
        value = float(value or 0)
    number = float(value)
    if number != number:
        return number
    if number >= 0:
        return math.floor(number + 0.5)
    return math.ceil(number - 0.5)


def date_parse(text):
    if not isinstance(text, str) or not text:
        return float("nan")
    raw = text.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return float("nan")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp() * 1000


class JSObj(dict):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.index = 0
        self.tokens = self._lex()
        self.pos = 0

    def _lex(self):
        tokens = []
        i = 0
        src = self.source
        n = len(src)
        punct = {
            "==": "EQ",
            "!=": "NE",
            "<=": "LE",
            ">=": "GE",
            "&&": "AND",
            "||": "OR",
        }
        while i < n:
            ch = src[i]
            if ch.isspace():
                i += 1
                continue
            if ch == "/" and i + 1 < n and src[i + 1] == "/":
                i = src.find("\n", i)
                if i < 0:
                    break
                continue
            if ch == "/" and i + 1 < n and src[i + 1] == "*":
                end = src.find("*/", i + 2)
                i = n if end < 0 else end + 2
                continue
            if ch in "\"'":
                quote = ch
                i += 1
                buf = []
                while i < n and src[i] != quote:
                    if src[i] == "\\":
                        buf.append(src[i + 1])
                        i += 2
                        continue
                    buf.append(src[i])
                    i += 1
                i += 1
                tokens.append(("STR", "".join(buf)))
                continue
            if ch.isdigit():
                start = i
                i += 1
                while i < n and (src[i].isdigit() or src[i] == "."):
                    i += 1
                tokens.append(("NUM", src[start:i]))
                continue
            two = src[i : i + 2]
            if two in punct:
                tokens.append((punct[two], two))
                i += 2
                continue
            if ch in "(){}[].,;:+-*/<>!=":
                tokens.append((ch, ch))
                i += 1
                continue
            if ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < n and (src[i].isalnum() or src[i] == "_"):
                    i += 1
                word = src[start:i]
                tokens.append(("ID", word))
                continue
            raise ParseError(f"bad character {ch!r} at {i}")
        tokens.append(("EOF", ""))
        return tokens

    def peek(self):
        return self.tokens[self.pos]

    def accept(self, kind):
        if self.peek()[0] == kind:
            token = self.peek()
            self.pos += 1
            return token
        return None

    def expect(self, kind):
        token = self.accept(kind)
        if token is None:
            raise ParseError(f"expected {kind} got {self.peek()}")
        return token

    def parse(self):
        items = []
        while self.peek()[0] != "EOF":
            if self.peek() == ("ID", "function"):
                items.append(self.parse_function())
            elif self.peek() == ("ID", "var"):
                items.append(self.parse_var())
            else:
                raise ParseError(f"top level {self.peek()}")
        return items

    def parse_function(self):
        self.expect("ID")
        name = self.expect("ID")[1]
        self.expect("(")
        params = []
        if not self.accept(")"):
            params.append(self.expect("ID")[1])
            while self.accept(","):
                params.append(self.expect("ID")[1])
            self.expect(")")
        body = self.parse_block()
        return ("fn", name, params, body)

    def parse_var(self):
        self.expect("ID")
        name = self.expect("ID")[1]
        self.expect("=")
        expr = self.parse_expr()
        self.expect(";")
        return ("var", name, expr)

    def parse_block(self):
        self.expect("{")
        stmts = []
        while not self.accept("}"):
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        token = self.peek()
        if token == ("ID", "if"):
            return self.parse_if()
        if token == ("ID", "while"):
            self.pos += 1
            self.expect("(")
            cond = self.parse_expr()
            self.expect(")")
            body = self.parse_block() if self.peek()[0] == "{" else [self.parse_stmt()]
            return ("while", cond, body)
        if token == ("ID", "return"):
            self.pos += 1
            if self.accept(";"):
                return ("return", ("null",))
            expr = self.parse_expr()
            self.expect(";")
            return ("return", expr)
        if token == ("ID", "var"):
            return self.parse_var()
        if token == ("ID", "continue"):
            self.pos += 1
            self.expect(";")
            return ("continue",)
        expr = self.parse_expr()
        if self.accept("="):
            if expr[0] not in {"name", "index", "member"}:
                raise ParseError("bad assign")
            value = self.parse_expr()
            self.expect(";")
            return ("assign", expr, value)
        self.expect(";")
        return ("expr", expr)

    def parse_if(self):
        self.expect("ID")
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        then = self.parse_block() if self.peek()[0] == "{" else [self.parse_stmt()]
        else_ = None
        if self.peek() == ("ID", "else"):
            self.pos += 1
            if self.peek() == ("ID", "if"):
                else_ = [self.parse_if()]
            else:
                else_ = self.parse_block() if self.peek()[0] == "{" else [self.parse_stmt()]
        return ("if", cond, then, else_)

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.accept("OR"):
            left = ("bin", "||", left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.accept("AND"):
            left = ("bin", "&&", left, self.parse_not())
        return left

    def parse_not(self):
        if self.accept("!"):
            return ("not", self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self):
        left = self.parse_add()
        while self.peek()[0] in {"EQ", "NE", "<", ">", "LE", "GE"}:
            op = self.peek()[1]
            self.pos += 1
            left = ("bin", op, left, self.parse_add())
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek()[0] in {"+", "-"}:
            op = self.peek()[0]
            self.pos += 1
            left = ("bin", op, left, self.parse_mul())
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek()[0] in {"*", "/"}:
            op = self.peek()[0]
            self.pos += 1
            left = ("bin", op, left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.accept("-"):
            return ("neg", self.parse_unary())
        if self.peek() == ("ID", "typeof"):
            self.pos += 1
            return ("typeof", self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if self.accept("."):
                name = self.expect("ID")[1]
                if self.accept("("):
                    args = self.parse_args()
                    expr = ("call", ("member", expr, name), args)
                else:
                    expr = ("member", expr, name)
                continue
            if self.accept("["):
                index = self.parse_expr()
                self.expect("]")
                expr = ("index", expr, index)
                continue
            if self.accept("("):
                args = self.parse_args()
                expr = ("call", expr, args)
                continue
            return expr

    def parse_args(self):
        args = []
        if self.accept(")"):
            return args
        args.append(self.parse_expr())
        while self.accept(","):
            args.append(self.parse_expr())
        self.expect(")")
        return args

    def parse_primary(self):
        token = self.peek()
        if token[0] == "NUM":
            self.pos += 1
            return ("num", token[1])
        if token[0] == "STR":
            self.pos += 1
            return ("str", token[1])
        if token == ("ID", "null"):
            self.pos += 1
            return ("null",)
        if token == ("ID", "true"):
            self.pos += 1
            return ("true",)
        if token == ("ID", "false"):
            self.pos += 1
            return ("false",)
        if self.accept("("):
            expr = self.parse_expr()
            self.expect(")")
            return expr
        if self.accept("{"):
            return self.parse_object()
        if self.accept("["):
            return self.parse_array()
        if token[0] == "ID":
            self.pos += 1
            return ("name", token[1])
        raise ParseError(f"expr {token}")

    def parse_object(self):
        fields = []
        if self.accept("}"):
            return ("obj", fields)
        fields.append(self.parse_field())
        while self.accept(","):
            if self.peek()[0] == "}":
                break
            fields.append(self.parse_field())
        self.expect("}")
        return ("obj", fields)

    def parse_field(self):
        token = self.peek()
        if token[0] == "STR":
            key = token[1]
            self.pos += 1
        else:
            key = self.expect("ID")[1]
        self.expect(":")
        return (key, self.parse_expr())

    def parse_array(self):
        items = []
        if self.accept("]"):
            return ("arr", items)
        items.append(self.parse_expr())
        while self.accept(","):
            if self.peek()[0] == "]":
                break
            items.append(self.parse_expr())
        self.expect("]")
        return ("arr", items)


def emit_expr(expr) -> str:
    kind = expr[0]
    if kind == "num":
        return expr[1]
    if kind == "str":
        return repr(expr[1])
    if kind == "null":
        return "None"
    if kind == "true":
        return "True"
    if kind == "false":
        return "False"
    if kind == "name":
        return expr[1]
    if kind == "not":
        return f"(not {emit_expr(expr[1])})"
    if kind == "neg":
        return f"(-({emit_expr(expr[1])}))"
    if kind == "typeof":
        return f"js_typeof({emit_expr(expr[1])})"
    if kind == "bin":
        op, left, right = expr[1], emit_expr(expr[2]), emit_expr(expr[3])
        if op == "+":
            return f"js_add({left}, {right})"
        py = {"-": "-", "*": "*", "/": "/", "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=", "||": "or", "&&": "and"}
        return f"(({left}) {py[op]} ({right}))"
    if kind == "member":
        obj, name = emit_expr(expr[1]), expr[2]
        if name == "length":
            return f"len({obj})"
        return f"getattr({obj}, {name!r})"
    if kind == "index":
        return f"js_index({emit_expr(expr[1])}, {emit_expr(expr[2])})"
    if kind == "call":
        callee, args = expr[1], expr[2]
        rendered = ", ".join(emit_expr(arg) for arg in args)
        if callee[0] == "member":
            obj = emit_expr(callee[1])
            name = callee[2]
            if name == "indexOf":
                return f"({obj}).find({rendered})"
            if name == "substring":
                start, end = (emit_expr(arg) for arg in args)
                return f"({obj})[{start}:{end}]"
            if name == "toLowerCase":
                return f"({obj}).lower()"
            if name == "parse" and callee[1] == ("name", "Date"):
                return f"date_parse({rendered})"
            if name == "round" and callee[1] == ("name", "Math"):
                return f"js_round({rendered})"
            if name == "max" and callee[1] == ("name", "Math"):
                return f"max({rendered})"
        return f"{emit_expr(callee)}({rendered})"
    if kind == "obj":
        fields = ", ".join(f"{key!r}: {emit_expr(value)}" for key, value in expr[1])
        return f"JSObj({{{fields}}})"
    if kind == "arr":
        return "[" + ", ".join(emit_expr(item) for item in expr[1]) + "]"
    raise ParseError(f"emit {kind}")


def emit_stmt(stmt, indent: int) -> list[str]:
    pad = "    " * indent
    kind = stmt[0]
    if kind == "var":
        return [f"{pad}{stmt[1]} = {emit_expr(stmt[2])}"]
    if kind == "assign":
        return [f"{pad}{emit_expr(stmt[1])} = {emit_expr(stmt[2])}"]
    if kind == "return":
        return [f"{pad}return {emit_expr(stmt[1])}"]
    if kind == "continue":
        return [f"{pad}continue"]
    if kind == "expr":
        return [f"{pad}{emit_expr(stmt[1])}"]
    if kind == "while":
        lines = [f"{pad}while {emit_expr(stmt[1])}:"]
        lines.extend(emit_body(stmt[2], indent + 1))
        return lines
    if kind == "if":
        lines = [f"{pad}if {emit_expr(stmt[1])}:"]
        lines.extend(emit_body(stmt[2], indent + 1))
        if stmt[3]:
            if len(stmt[3]) == 1 and stmt[3][0][0] == "if":
                nested = emit_stmt(stmt[3][0], indent)
                lines.append(f"{pad}el{nested[0].lstrip()}")
                lines.extend(nested[1:])
            else:
                lines.append(f"{pad}else:")
                lines.extend(emit_body(stmt[3], indent + 1))
        return lines
    raise ParseError(f"stmt {kind}")


def emit_body(stmts, indent: int) -> list[str]:
    if not stmts:
        return ["    " * indent + "pass"]
    lines = []
    for stmt in stmts:
        lines.extend(emit_stmt(stmt, indent))
    return lines


def translate(source: str) -> str:
    program = Parser(source).parse()
    lines = []
    for item in program:
        if item[0] == "var":
            lines.extend(emit_stmt(item, 0))
            continue
        _, name, params, body = item
        lines.append(f"def {name}({', '.join(params)}):")
        lines.extend(emit_body(body, 1))
        lines.append("")
    return "\n".join(lines) + "\n"


HELPERS = """
class JSObj(dict):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
"""


_MARK = re.compile(r"/\* cortex-measure:start \*/\n(.*)\n[ \t]*/\* cortex-measure:end \*/", re.S)
_NOW = "2026-11-27T12:00:00Z"
_BARE_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def measure_source(text: str) -> str:
    match = _MARK.search(text)
    if match is None:
        raise AssertionError("measure block is missing")
    lines = []
    for line in match.group(1).splitlines():
        if line.startswith("    "):
            line = line[4:]
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def state_map_keys(text: str) -> list[str]:
    match = re.search(r"const STATE_MAP = \{([^}]+)\}", text)
    if match is None:
        raise AssertionError("STATE_MAP is missing")
    return re.findall(r"(\w+)\s*:", match.group(1))


class _NumericElements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._stack: list[dict[str, object]] = []
        self.mapped: list[tuple[str | None, str]] = []
        self.bare: list[tuple[str, str]] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name: value or "" for name, value in attrs}
        if attr.get("id"):
            self.ids.append(str(attr["id"]))
        self._stack.append({"id": attr.get("id"), "key": attr.get("data-state-map"), "text": []})

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        item = self._stack.pop()
        text = "".join(item["text"]).strip()
        if item["key"]:
            self.mapped.append((None if item["id"] is None else str(item["id"]), str(item["key"])))
        element_id = item["id"]
        if element_id and _BARE_NUMBER.fullmatch(text):
            self.bare.append((str(element_id), text))

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1]["text"].append(data)


def _load_measure(source: str) -> dict[str, object]:
    namespace: dict[str, object] = {
        "JSObj": JSObj,
        "js_add": js_add,
        "js_typeof": js_typeof,
        "js_round": js_round,
        "js_index": js_index,
        "date_parse": date_parse,
        "math": math,
    }
    exec(translate(source), namespace)
    return namespace


def _wrap(namespace: dict[str, object], value: object) -> object:
    factory = namespace["JSObj"]
    if isinstance(value, dict):
        return factory({key: _wrap(namespace, item) for key, item in value.items()})
    return value


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class TruthfulStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pulse = (ROOT / "pulse.html").read_text(encoding="utf-8")
        cls.html = (ROOT / "dashboard.html").read_text(encoding="utf-8")
        cls.script = (ROOT / "dashboard.js").read_text(encoding="utf-8")
        cls.measure = measure_source(cls.pulse)
        cls.runtime = _load_measure(cls.measure)
        cls.now = cls.runtime["date_parse"](_NOW)

    def test_both_surfaces_share_one_measure_block(self) -> None:
        self.assertEqual(self.measure, measure_source(self.script))
        self.assertEqual(self.runtime["COUNT_STALE_HOURS"], 24)
        self.assertEqual(self.runtime["DREAM_FRESH_HOURS"], 36)
        for name in ("renderMeasure", "memoryFromCounts", "dreamFromPulse", "dreamFromRun", "countOrAbsent"):
            self.assertIn(name, self.runtime)

    def test_every_numeric_element_has_a_state_function(self) -> None:
        pulse_keys = set(state_map_keys(self.pulse))
        dash_keys = set(state_map_keys(self.script))
        pulse_dom = _NumericElements()
        pulse_dom.feed(self.pulse)
        dash_dom = _NumericElements()
        dash_dom.feed(self.html)
        self.assertFalse(pulse_dom.bare, pulse_dom.bare)
        self.assertFalse(dash_dom.bare, dash_dom.bare)
        self.assertGreaterEqual(len(pulse_dom.mapped), 3)
        mapped_ids = [element_id for element_id, _key in dash_dom.mapped]
        self.assertEqual(len(mapped_ids), len(set(mapped_ids)))
        for element_id, key in pulse_dom.mapped:
            self.assertIn(key, pulse_keys, element_id)
            self.assertIn(f"function map{key[:1].upper()}{key[1:]}", self.pulse)
        for element_id, key in dash_dom.mapped:
            self.assertIn(key, dash_keys, element_id)
            self.assertIn(f"function map{key[:1].upper()}{key[1:]}", self.script)
        self.assertIn("count", dash_keys)
        self.assertIn("dream", dash_keys)
        self.assertIn('dataset.stateMap=value&&value.kind==="dream"?"dream":"count"', self.script)
        painted = set(re.findall(r'paintMeasured\("([^"]+)"', self.script))
        self.assertTrue(painted)
        self.assertTrue(painted <= set(dash_dom.ids))
        for element_id in ("pulse-memories", "pulse-recalled", "pulse-dream"):
            self.assertIn(element_id, pulse_dom.ids)

    def test_null_and_zero_counts_are_not_a_healthy_zero(self) -> None:
        render = self.runtime["renderMeasure"]
        absent = _plain(render(_wrap(self.runtime, {"kind": "count", "value": None})))
        zero = _plain(render(_wrap(self.runtime, {"kind": "count", "value": 0, "zeroIsAbsent": True})))
        recalled = _plain(render(self.runtime["countOrAbsent"](0)))
        fresh = _plain(render(_wrap(self.runtime, {"kind": "count", "value": 12})))
        empty = _plain(render(_wrap(self.runtime, {"kind": "count", "emptyStore": True, "value": 0})))
        self.assertEqual(absent, {"state": "no_data", "text": "not measurable yet"})
        self.assertEqual(zero["text"], "not measurable yet")
        self.assertEqual(recalled["text"], "not measurable yet")
        self.assertEqual(fresh, {"state": "fresh", "text": "12 ✓"})
        self.assertEqual(empty["text"], "0")
        self.assertEqual(empty["note"], "the store is empty")
        self.assertNotEqual(empty["text"], fresh["text"])
        memories = _plain(self.runtime["memoryFromCounts"](_wrap(self.runtime, {"active": 0, "cold": 0, "archived": 0}), "the store is empty"))
        self.assertTrue(memories["emptyStore"])
        present = _plain(self.runtime["memoryFromCounts"](_wrap(self.runtime, {"active": 2, "cold": 0, "archived": 1}), None))
        self.assertEqual(present["value"], 3)
        missing = _plain(self.runtime["memoryFromCounts"](None, None))
        self.assertIsNone(missing["value"])

    def test_four_states_on_each_surface(self) -> None:
        render = self.runtime["renderMeasure"]
        now = self.now
        dream = self.runtime["dreamFromPulse"]
        run = self.runtime["dreamFromRun"]
        cases = {
            "no_data": render(dream(None, now)),
            "running": render(dream(_wrap(self.runtime, {"state": "running", "progress": "40%"}), now)),
            "stale": render(dream(_wrap(self.runtime, {"state": "stale", "at": "2026-11-26T12:00:00Z"}), now)),
            "fresh": render(dream(_wrap(self.runtime, {"state": "fresh", "at": "2026-11-27T04:15:00Z"}), now)),
        }
        self.assertEqual(_plain(cases["no_data"])["text"], "not run yet")
        self.assertEqual(_plain(cases["running"])["text"], "running… 40%")
        self.assertEqual(_plain(cases["stale"])["text"], "1 d ago ⚠")
        self.assertEqual(_plain(cases["fresh"])["text"], "04:15 ✓")
        failed = _plain(render(dream(_wrap(self.runtime, {"state": "failed", "at": "2026-11-27T04:15:00Z"}), now)))
        self.assertEqual(failed["text"], "failed 04:15")
        running = _plain(run(_wrap(self.runtime, {"status": "running", "episodes_replayed": 0, "applied_changes": 0}), now, _wrap(self.runtime, {"status": "running", "progress": 40})))
        self.assertEqual(running["state"], "running")
        self.assertNotIn("applied_changes", running)
        self.assertNotIn("episodes_replayed", running)
        fresh_run = _plain(render(run(_wrap(self.runtime, {"status": "completed", "completed_at": "2026-11-27T04:15:00Z"}), now, None)))
        stale_run = _plain(render(run(_wrap(self.runtime, {"status": "completed", "completed_at": "2026-11-25T12:00:00Z"}), now, None)))
        self.assertEqual(fresh_run["text"], "04:15 ✓")
        self.assertEqual(stale_run["state"], "stale")
        self.assertIn("⚠", stale_run["text"])
        aged = self.runtime["withFreshness"](_wrap(self.runtime, {"kind": "count", "value": 7}), "2026-11-26T12:00:00Z", now, 24)
        self.assertEqual(_plain(render(aged))["text"], "7 · 1 d ago ⚠")
        recent = _plain(self.runtime["withFreshness"](_wrap(self.runtime, {"kind": "count", "value": 7}), "2026-11-27T11:00:00Z", now, 24))
        self.assertNotIn("stale", recent)
        surfaces = (self.pulse, self.script)
        for surface in surfaces:
            self.assertIn("not measurable yet", surface)
            self.assertIn("not run yet", surface)
            self.assertIn("running…", surface)
            self.assertIn("⚠", surface)
            self.assertIn("✓", surface)
            self.assertIn("the store is empty", surface)
            self.assertIn("server unreachable", surface)
            self.assertIn("skeleton-row", surface if surface is self.pulse else self.html)

    def test_empty_store_server_down_and_loading_are_distinct(self) -> None:
        self.assertIn('id="empty-store"', self.pulse)
        self.assertIn('id="empty-store"', self.html)
        self.assertIn('id="server-banner"', self.html)
        self.assertIn('hold(UNREACHABLE)', self.pulse)
        self.assertIn('holdSnapshot("server unreachable")', self.script)
        self.assertGreaterEqual(self.pulse.count("skeleton-row"), 3)
        self.assertGreaterEqual(self.html.count("skeleton-row"), 12)
        self.assertIn('return "not measurable yet"', self.script)
        for surface in (self.pulse, self.html, self.script):
            self.assertNotIn("INSERT INTO meta", surface)
            self.assertNotIn("cdn.jsdelivr.net", surface)
            self.assertNotIn("node_modules", surface)

    def test_node_agrees_when_node_is_installed(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        script = r"""
const fs = require("fs");
const vm = require("vm");
const code = fs.readFileSync(process.argv[1], "utf8");
const context = { Math, Date };
vm.createContext(context);
vm.runInContext(code, context);
const now = Date.parse("2026-11-27T12:00:00Z");
const out = {
  absent: context.renderMeasure({kind:"count", value:null}),
  empty: context.renderMeasure({kind:"count", emptyStore:true, value:0}),
  fresh: context.renderMeasure({kind:"count", value:12}),
  stale: context.renderMeasure(context.withFreshness({kind:"count", value:7}, "2026-11-26T12:00:00Z", now, 24)),
  dream: context.renderMeasure(context.dreamFromPulse({state:"fresh", at:"2026-11-27T04:15:00Z"}, now)),
  running: context.dreamFromRun({status:"running", applied_changes:0}, now, {status:"running", progress:40})
};
console.log(JSON.stringify(out));
"""
        source = self.measure
        temp = Path("/tmp/cortex-measure-check.js")
        temp.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run([node, "-e", script, str(temp)], check=True, capture_output=True, text=True)
        finally:
            temp.unlink(missing_ok=True)
        got = json.loads(completed.stdout)
        self.assertEqual(got["absent"]["text"], "not measurable yet")
        self.assertEqual(got["empty"]["note"], "the store is empty")
        self.assertEqual(got["fresh"]["text"], "12 ✓")
        self.assertEqual(got["stale"]["text"], "7 · 1 d ago ⚠")
        self.assertEqual(got["dream"]["text"], "04:15 ✓")
        self.assertEqual(got["running"]["state"], "running")
        self.assertNotIn("applied_changes", got["running"])


if __name__ == "__main__":
    unittest.main()
