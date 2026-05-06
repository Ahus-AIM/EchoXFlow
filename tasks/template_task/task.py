from __future__ import annotations

from typing import Any

from tasks.registry import TaskSpec


def loss_terms(model: Any, sample: object, config: object) -> dict[str, Any]:
    # TODO: Run the model, compute task-specific loss terms, and return at least {"loss": loss}.
    raise NotImplementedError("template_task loss is not implemented")


TASK_SPEC = TaskSpec(name="template_task", loss_fn=loss_terms)
