"""Model choice for the night (#118): the ladder is declared, the pin is authoritative, fallback is loud."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import model_choice

# The ladder's default key variable: a tier is usable only when the variable it names is present.
KEY = "OPENROUTER_API_KEY"


def env(**values: str) -> dict[str, str]:
    base = {KEY: "present"}
    base.update(values)
    return base


class ChoiceTests(unittest.TestCase):
    # -- the ladder -------------------------------------------------------------------

    def test_the_strongest_usable_tier_wins(self) -> None:
        choice = model_choice.choose(env=env(
            CORTEX_SLEEP_MODEL="strong/model",
            CORTEX_SLEEP_ENDPOINT="https://strong.example/v1",
            CORTEX_AUTO_JUDGE_MODEL="judge/model",
            CORTEX_AUTO_JUDGE_ENDPOINT="https://judge.example/v1",
        ))
        self.assertTrue(choice.usable)
        self.assertEqual(choice.tier, "sleep_explicit")
        self.assertEqual(choice.model, "strong/model")
        self.assertFalse(choice.fell_back)

    def test_a_missing_key_drops_the_tier_and_the_next_one_wins_loudly(self) -> None:
        choice = model_choice.choose(env={
            "CORTEX_SLEEP_MODEL": "strong/model",
            "CORTEX_SLEEP_ENDPOINT": "https://strong.example/v1",
            "CORTEX_AUTO_JUDGE_MODEL": "judge/model",
            "CORTEX_AUTO_JUDGE_ENDPOINT": "https://judge.example/v1",
            KEY: "present",
        } | {"CORTEX_SLEEP_API_KEY_ENV": "A_KEY_THAT_IS_NOT_SET"})
        self.assertTrue(choice.usable)
        self.assertEqual(choice.tier, "judge")
        self.assertTrue(choice.fell_back, "a cheap night must be visible, not look like a strong one")
        self.assertIn("stronger or earlier tier", choice.reason)
        self.assertEqual(choice.skipped[0]["tier"], "sleep_explicit")

    def test_the_skip_reason_names_the_variable_and_never_its_value(self) -> None:
        choice = model_choice.choose(env={
            "CORTEX_SLEEP_MODEL": "strong/model",
            "CORTEX_SLEEP_ENDPOINT": "https://strong.example/v1",
            "CORTEX_SLEEP_API_KEY_ENV": "MISSING_KEY",
        })
        rendered = str(choice.as_dict())
        self.assertIn("MISSING_KEY", rendered)
        self.assertNotIn("present", rendered.replace("MISSING_KEY", ""))

    def test_a_model_without_an_endpoint_is_not_a_candidate(self) -> None:
        choice = model_choice.choose(env=env(CORTEX_SLEEP_MODEL="strong/model"))
        self.assertFalse(choice.usable)
        self.assertIn("CORTEX_SLEEP_ENDPOINT", choice.reason)

    # -- the chat model is never inherited --------------------------------------------

    def test_no_tier_configured_means_no_model_rather_than_a_default(self) -> None:
        choice = model_choice.choose(env={})
        self.assertFalse(choice.usable)
        self.assertEqual(choice.model, "", "inventing a default model is how a store starts spending silently")
        self.assertIn("no tier is usable", choice.reason)
        self.assertIn("no model available", choice.render())

    def test_the_ladder_lists_only_deep_work_providers(self) -> None:
        names = {entry["tier"] for entry in model_choice.LADDER}
        self.assertEqual(names, {"sleep_explicit", "judge", "mechanics"})
        self.assertNotIn("chat", names)

    # -- the pin ----------------------------------------------------------------------

    def test_a_pin_is_used_exactly(self) -> None:
        choice = model_choice.choose(env=env(
            CORTEX_SLEEP_ENDPOINT="https://strong.example/v1",
        ), pin="pinned/model")
        self.assertEqual(choice.model, "pinned/model")
        self.assertTrue(choice.pinned)
        self.assertTrue(choice.usable)

    def test_a_pin_without_a_key_is_refused_rather_than_downgraded(self) -> None:
        choice = model_choice.choose(env={
            # No key anywhere: the pin is configured but cannot be called.
            "CORTEX_SLEEP_ENDPOINT": "https://strong.example/v1",
            "CORTEX_AUTO_JUDGE_MODEL": "judge/model",
            "CORTEX_AUTO_JUDGE_ENDPOINT": "https://judge.example/v1",
        }, pin="pinned/model")
        self.assertFalse(choice.usable, "silently using a weaker model for a pinned job is worse than failing")
        self.assertIn("refusing", choice.reason)
        self.assertIn("pinned/model", choice.reason)

    def test_an_env_model_is_ambient_config_not_a_pin(self) -> None:
        """A default that goes stale must not block the night; an explicit per-run pin must."""
        choice = model_choice.choose(env=env(
            CORTEX_SLEEP_MODEL="env/model",
            CORTEX_SLEEP_ENDPOINT="https://strong.example/v1",
        ))
        self.assertTrue(choice.usable)
        self.assertEqual(choice.model, "env/model")
        self.assertFalse(choice.pinned)

    def test_an_unpinned_fallback_is_not_reported_as_pinned(self) -> None:
        choice = model_choice.choose(env=env(
            CORTEX_AUTO_JUDGE_MODEL="judge/model",
            CORTEX_AUTO_JUDGE_ENDPOINT="https://judge.example/v1",
        ))
        self.assertTrue(choice.usable)
        self.assertFalse(choice.pinned)

    # -- the record -------------------------------------------------------------------

    def test_the_choice_is_recordable_without_a_key(self) -> None:
        choice = model_choice.choose(env=env(
            CORTEX_SLEEP_MODEL="strong/model",
            CORTEX_SLEEP_ENDPOINT="https://strong.example/v1",
        ))
        record = choice.as_dict()
        self.assertEqual(record["model"], "strong/model")
        self.assertEqual(record["api_key_env"], "OPENROUTER_API_KEY")
        self.assertNotIn("sk-", str(record), "the record names the variable, never the secret")

    def test_the_render_names_the_tier_and_the_key_variable(self) -> None:
        choice = model_choice.choose(env=env(
            CORTEX_SLEEP_MODEL="strong/model",
            CORTEX_SLEEP_ENDPOINT="https://strong.example/v1",
        ))
        rendered = choice.render()
        self.assertIn("sleep_explicit", rendered)
        self.assertIn("OPENROUTER_API_KEY", rendered)

    def test_available_is_a_plain_answer(self) -> None:
        self.assertFalse(model_choice.available({}))
        self.assertTrue(model_choice.available(env(
            CORTEX_SLEEP_MODEL="m", CORTEX_SLEEP_ENDPOINT="https://x.example/v1")))

    # -- it spends nothing ------------------------------------------------------------

    def test_resolving_does_not_touch_the_environment(self) -> None:
        before = dict(__import__("os").environ)
        model_choice.choose(env={})
        self.assertEqual(before, dict(__import__("os").environ))


if __name__ == "__main__":
    unittest.main()
