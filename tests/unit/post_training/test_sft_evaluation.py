from pathlib import Path

import yaml

from atomllm.post_training.sft_evaluation import _load_config


def test_sft_evaluation_config_loads_fixed_checkpoint_contract(
    tmp_path: Path,
) -> None:
    config = {
        "schema_version": 1,
        "evaluation_id": "heldout-v1",
        "model_config": "model.yaml",
        "model_config_sha256": "a" * 64,
        "heldout_data": "heldout",
        "heldout_manifest_sha256": "b" * 64,
        "micro_batch_size": 2,
        "loss_chunk_size": 1024,
        "expected_world_size": 6,
        "checkpoints": [
            {
                "name": "candidate",
                "path": "checkpoint",
                "manifest_sha256": "c" * 64,
                "model_sha256": "d" * 64,
            }
        ],
    }
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = _load_config(path)

    assert loaded["evaluation_id"] == "heldout-v1"
    assert loaded["expected_world_size"] == 6
    assert loaded["checkpoints"][0]["name"] == "candidate"
