"""Regression tests for selection-stage token memoisation (2026-09-10).

`_memory_similarity` is called O(candidates x selected) times per recall, and
always with the same small set of memory bodies. It used to re-tokenize BOTH
bodies on every call, which dominated live prepare latency — profiled at 95% of
the select stage for a single recall (5,886 `diversified_value` calls, 27,484
`_memory_similarity` calls, 55,501 tokenizer runs, ~3.9M `casefold` calls).

The fix memoises the token set per body. Two properties matter and both are
pinned here: the similarity VALUE is unchanged, and the tokenizer runs once per
distinct body instead of once per comparison.
"""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import retrieval
from cortex.retrieval import (
    RetrievalResult,
    _memory_similarity,
)
from cortex.store import query_tokens


def token_cache_clear() -> None:
    """Clear the memo cache if the helper exists (it does not on pre-change code)."""
    helper = getattr(retrieval, "_content_token_set", None)
    if helper is not None:
        helper.cache_clear()


def make_result(content: str, *, memory_id: str = "m1") -> RetrievalResult:
    return RetrievalResult(
        memory={"id": memory_id, "content": content, "kind": "semantic", "state": "active"},
        score=0.5,
        components={},
        estimated_tokens=10,
    )


def uncached_similarity(left: RetrievalResult, right: RetrievalResult) -> float:
    """The pre-memoisation definition, kept here as the reference."""
    a = set(query_tokens(left.memory["content"]))
    b = set(query_tokens(right.memory["content"]))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SelectTokenMemoTests(unittest.TestCase):
    def setUp(self) -> None:
        token_cache_clear()

    def tearDown(self) -> None:
        token_cache_clear()

    # -- the value must not change --------------------------------------

    def test_similarity_matches_the_uncached_definition(self) -> None:
        cases = [
            ("alpha beta gamma delta", "alpha beta epsilon"),
            ("the homelab tunnel runs on port 8000", "the tunnel port is 8000"),
            ("completely different words here", "nothing at all in common"),
            ("exact duplicate body", "exact duplicate body"),
            ("a", "a b c d e f"),
        ]
        for left_content, right_content in cases:
            with self.subTest(left=left_content, right=right_content):
                left = make_result(left_content, memory_id="l")
                right = make_result(right_content, memory_id="r")
                self.assertAlmostEqual(
                    _memory_similarity(left, right),
                    uncached_similarity(left, right),
                    places=12,
                )

    def test_repeated_calls_return_the_same_value(self) -> None:
        left = make_result("alpha beta gamma delta", memory_id="l")
        right = make_result("alpha beta epsilon", memory_id="r")
        first = _memory_similarity(left, right)
        for _ in range(5):
            self.assertEqual(_memory_similarity(left, right), first)

    def test_empty_and_whitespace_bodies_return_zero(self) -> None:
        empty = make_result("", memory_id="e")
        blanks = make_result("   \n\t  ", memory_id="w")
        real = make_result("alpha beta", memory_id="r")
        for other in (empty, blanks):
            with self.subTest(content=repr(other.memory["content"])):
                self.assertEqual(_memory_similarity(real, other), 0.0)
                self.assertEqual(_memory_similarity(other, real), 0.0)

    # -- the point of the change: tokenize once per body ------------------

    def test_tokenizer_runs_once_per_distinct_body(self) -> None:
        calls: list[str] = []
        original = retrieval.query_tokens

        def counting(text: str) -> list[str]:
            calls.append(text)
            return original(text)

        retrieval.query_tokens = counting  # type: ignore[assignment]
        try:
            token_cache_clear()
            left = make_result("unique-left-body-for-memo-test", memory_id="l")
            right = make_result("unique-right-body-for-memo-test", memory_id="r")
            for _ in range(20):
                _memory_similarity(left, right)
            # 20 comparisons, 2 distinct bodies -> exactly 2 tokenizations.
            self.assertEqual(len(calls), 2, f"tokenizer ran {len(calls)} times")
        finally:
            retrieval.query_tokens = original  # type: ignore[assignment]

    def test_cache_is_bounded(self) -> None:
        helper = getattr(retrieval, "_content_token_set", None)
        if helper is None:
            self.fail(
                "token memoisation helper is missing — the selection stage would "
                "re-tokenize every body on every comparison"
            )
        info = helper.cache_info()
        self.assertIsNotNone(
            info.maxsize, "an unbounded cache would grow without limit on live traffic"
        )
        self.assertGreater(info.maxsize or 0, 0)


if __name__ == "__main__":
    unittest.main()
