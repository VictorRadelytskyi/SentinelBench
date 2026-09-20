import pytest

from sentinelbench.data import PreferenceExample, Rollout, SFTExample


def test_training_records_are_structured_and_validated() -> None:
    record = SFTExample("system", "investigate", "contained host")
    preference = PreferenceExample("task", "evidence first", "act blindly", "safety")
    rollout = Rollout("task", "answer", 1.0, ("lookup_alert",))

    assert record.messages()[-1]["role"] == "assistant"
    preference.validate()
    rollout.validate()


def test_preference_rejects_duplicate_completions() -> None:
    with pytest.raises(ValueError, match="must differ"):
        PreferenceExample("task", "same", "same", "quality").validate()