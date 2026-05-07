from __future__ import annotations

import pytest

from tasks.registry import _task_module_dirs, available_task_names, task_train_yaml


def test_available_task_names_excludes_template_scaffold() -> None:
    assert "_template_task" not in available_task_names()


def test_task_module_dirs_excludes_underscore_prefixed_tasks() -> None:
    assert all(not task_dir.name.startswith("_") for task_dir in _task_module_dirs())


def test_template_scaffold_is_not_supported_task() -> None:
    with pytest.raises(ValueError, match=r"Unsupported task '_template_task'"):
        task_train_yaml("_template_task")
