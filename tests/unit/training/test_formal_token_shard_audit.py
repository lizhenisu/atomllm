from pathlib import Path

from atomllm.training.formal_token_shard_audit import (
    load_formal_token_shard_audit_config,
)


CONFIG = Path("configs/training/formal-token-shard-audit-v2.yaml")


def test_loads_formal_token_shard_audit_contract() -> None:
    config = load_formal_token_shard_audit_config(CONFIG)

    assert config.sequence_length == 1024
    assert config.batch_size == 5
    assert config.resume_probe_batches == 3
