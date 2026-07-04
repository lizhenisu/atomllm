from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from atomllm.model.config import calculate_parameter_count, load_model_config
from atomllm.model.model import AtomLLM


CONFIG_PATH = Path("configs/model/atom-base-300m.yaml")


def small_model_config():
    config = load_model_config(CONFIG_PATH)
    return replace(
        config,
        tokenizer=replace(config.tokenizer, vocab_size=512),
        dimensions=replace(
            config.dimensions,
            max_sequence_length=32,
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            ffn_hidden_size=64,
        ),
        expected_parameter_count=34_976,
    )


def make_model() -> AtomLLM:
    torch.manual_seed(37)
    return AtomLLM(small_model_config()).eval()


def test_model_output_shape_loss_and_backward() -> None:
    model = make_model().train()
    input_ids = torch.randint(4, 512, (2, 7))

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 7, 512)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.past_key_values is None
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_lm_head_and_embeddings_share_the_same_parameter() -> None:
    model = make_model()

    assert model.lm_head.weight is model.token_embeddings.weight
    model = model.to(dtype=torch.bfloat16)
    assert model.lm_head.weight is model.token_embeddings.weight
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_small_model_real_parameter_count_matches_formula() -> None:
    config = small_model_config()
    model = AtomLLM(config)

    real_count = sum(parameter.numel() for parameter in model.parameters())
    calculated_count = calculate_parameter_count(config).total

    assert real_count == 34_976
    assert real_count == calculated_count


def test_full_model_meta_parameter_count_is_exact() -> None:
    config = load_model_config(CONFIG_PATH)

    with torch.device("meta"):
        model = AtomLLM(config)

    real_count = sum(parameter.numel() for parameter in model.parameters())

    assert real_count == 303_350_784
    assert model.lm_head.weight is model.token_embeddings.weight


def test_next_token_loss_matches_manual_cross_entropy() -> None:
    model = make_model()
    input_ids = torch.tensor([[2, 10, 11, 12, 3]])

    output = model(input_ids, labels=input_ids)
    expected = functional.cross_entropy(
        output.logits[:, :-1].float().reshape(-1, 512),
        input_ids[:, 1:].reshape(-1),
    )

    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected)


def test_loss_ignores_pad_and_attention_mask_targets() -> None:
    model = make_model()
    input_ids = torch.tensor([[2, 20, 0, 30, 31]])
    labels = input_ids.clone()
    attention_mask = torch.tensor([[1, 1, 1, 0, 1]])

    output = model(
        input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )
    effective_labels = labels.clone()
    effective_labels[effective_labels == 0] = -100
    effective_labels[attention_mask == 0] = -100
    expected = functional.cross_entropy(
        output.logits[:, :-1].float().reshape(-1, 512),
        effective_labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected)


def test_model_causality_blocks_future_token_changes() -> None:
    model = make_model()
    original = torch.randint(4, 512, (1, 7))
    changed = original.clone()
    changed[:, 5:] = torch.randint(4, 512, (1, 2))

    original_logits = model(original).logits
    changed_logits = model(changed).logits

    torch.testing.assert_close(
        original_logits[:, :5],
        changed_logits[:, :5],
        rtol=1e-5,
        atol=1e-5,
    )


def test_multilayer_incremental_cache_matches_full_logits() -> None:
    model = make_model()
    input_ids = torch.randint(4, 512, (1, 8))

    full_logits = model(input_ids).logits
    cache = None
    pieces = []
    for index in range(input_ids.shape[1]):
        output = model(
            input_ids[:, index : index + 1],
            attention_mask=torch.ones(1, index + 1, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
        )
        pieces.append(output.logits)
        cache = output.past_key_values

    torch.testing.assert_close(
        torch.cat(pieces, dim=1),
        full_logits,
        rtol=1e-5,
        atol=1e-5,
    )
    assert cache is not None and len(cache) == 2
    assert all(layer_cache.key.shape == (1, 2, 8, 8) for layer_cache in cache)


def test_model_rejects_invalid_inputs_labels_and_cache() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="input_ids must be in"):
        model(torch.tensor([[512]]))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(torch.ones(1, 2))
    with pytest.raises(ValueError, match="same.*shape"):
        model(torch.tensor([[2, 3]]), labels=torch.tensor([[2]]))
    with pytest.raises(ValueError, match="must contain 2"):
        model(torch.tensor([[2, 3]]), past_key_values=tuple())


def test_model_bfloat16_cpu_forward_backward_and_cache() -> None:
    model = make_model().to(dtype=torch.bfloat16).train()
    input_ids = torch.randint(4, 512, (1, 5))

    output = model(input_ids, labels=input_ids, use_cache=True)

    assert output.logits.dtype == torch.bfloat16
    assert output.loss is not None
    assert output.past_key_values is not None
    output.loss.backward()
    assert model.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.token_embeddings.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_small_model_cuda_bfloat16_forward_backward_and_cache() -> None:
    model = make_model().to(device="cuda", dtype=torch.bfloat16).train()
    input_ids = torch.randint(4, 512, (1, 6), device="cuda")

    output = model(input_ids, labels=input_ids, use_cache=True)

    assert output.logits.device.type == "cuda"
    assert output.logits.dtype == torch.bfloat16
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.past_key_values is not None
    output.loss.backward()
    assert model.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.token_embeddings.weight.grad).all()
