from pathlib import Path

import pytest
import yaml

from atomllm.model.config import load_model_config
from atomllm.training.config import load_training_config
from atomllm.training.context_audit import (
    ContextAuditError,
    load_context_audit_config,
)


CONFIG = Path("configs/training/atom-base-300m-long-audit.yaml")


def test_final_context_configs_freeze_40960_tokens() -> None:
    audit = load_context_audit_config(CONFIG)
    formal = load_training_config(audit.formal_config)
    smoke = load_training_config(audit.smoke_config)
    model = load_model_config(formal.model.config_path)

    assert audit.expected_sequence_length == 40_960
    assert audit.baseline_sequence_length == 27_136
    assert model.expected_parameter_count == 303_350_784
    assert model.dimensions.max_sequence_length == 40_960
    assert formal.batch.sequence_length == 40_960
    assert formal.batch.tokens_per_optimizer_step == 40_960
    assert formal.runtime.gradient_checkpointing
    assert formal.runtime.checkpoint_segment_layers == 4
    assert formal.runtime.loss_chunk_size == 256
    assert smoke.scheduler.total_steps == 2


def test_context_window_audit_rejects_invalid_limit(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["maximum_peak_allocated_gib"] = 0
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ContextAuditError, match="must be positive"):
        load_context_audit_config(path)
