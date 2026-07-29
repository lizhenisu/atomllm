from atomllm.training.base_probe import PROBES


def test_base_probes_have_unique_names_and_one_expected_candidate() -> None:
    names = [name for name, *_ in PROBES]

    assert len(names) == len(set(names))
    assert all(expected in candidates for _, _, candidates, expected in PROBES)
