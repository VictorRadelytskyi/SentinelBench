from sentinelbench.reporting import objective_sanity_report, project_report


def test_project_report_is_machine_readable_and_complete() -> None:
    report = project_report()

    assert report["readiness"]["core_algorithms"] is True
    assert report["data_inventory"]["sft_examples"] == 2
    assert report["stack"]["transformers"]["installed"] is True
    assert report["runtime"]["torch"]


def test_objective_sanity_metrics_are_monotonic_for_preference_tasks() -> None:
    sanity = objective_sanity_report()

    assert sanity["dpo"]["preference_accuracy"] == 1.0
    assert sanity["reward_model"]["preference_accuracy"] == 1.0
    assert sanity["ppo"]["preference_accuracy"] == 1.0
    assert sanity["grpo"]["preference_accuracy"] >= 0.66