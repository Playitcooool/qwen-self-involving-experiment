import json

import torch

from lopd_lite import Experience, ExperienceBank, LatentComposer, completion_slices, reverse_kl_topk


def test_completion_slices_align_causal_positions():
    student, teacher = completion_slices(prompt_len=5, completion_len=3, latent_len=4)
    assert (student.start, student.stop) == (4, 7)
    assert (teacher.start, teacher.stop) == (8, 11)


def test_reverse_kl_is_zero_for_identical_logits():
    logits = torch.randn(1, 3, 17)
    value = reverse_kl_topk(logits, logits.clone(), top_k=5)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-5)


def test_reverse_kl_propagates_teacher_gradient():
    student = torch.randn(1, 2, 13, requires_grad=True)
    teacher = torch.randn(1, 2, 13, requires_grad=True)
    loss = reverse_kl_topk(student, teacher, top_k=4).mean()
    loss.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    assert teacher.grad is not None and float(teacher.grad.abs().sum()) > 0


def test_composer_shape_and_gradient():
    composer = LatentComposer(hidden_size=12, latent_tokens=4)
    memory = torch.randn(1, 7, 12)
    out = composer(memory)
    assert out.shape == (1, 4, 12)
    out.square().mean().backward()
    assert float(composer.queries.grad.abs().sum()) > 0


def test_experience_bank_retrieval_excludes_current_task_and_serializes_without_gold():
    bank = ExperienceBank()
    bank.add(Experience("a", "x", "task alpha", "answer-like generated trajectory"), torch.tensor([1.0, 0.0]))
    bank.add(Experience("b", "x", "task beta", "another generated trajectory"), torch.tensor([0.9, 0.1]))
    found = bank.retrieve(torch.tensor([1.0, 0.0]), top_k=1, exclude_task_id="a")
    assert [item.task_id for item in found] == ["b"]


def test_experience_bank_does_not_fallback_to_excluded_task():
    bank = ExperienceBank()
    bank.add(Experience("only", "x", "task", "trajectory"), torch.ones(3))
    found = bank.retrieve(torch.ones(3), top_k=1, exclude_task_id="only")
    assert found == []


def test_experience_jsonl_has_no_answer_field(tmp_path):
    bank = ExperienceBank()
    bank.add(Experience("a", "x", "task", "trajectory"), torch.ones(3))
    path = tmp_path / "bank.jsonl"
    bank.write_jsonl(path)
    record = json.loads(path.read_text())
    assert "answer" not in record
