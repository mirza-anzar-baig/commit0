from __future__ import annotations

from unittest.mock import patch

import pytest

from commit0.harness.constants import RepoInstance, SimpleInstance
from commit0.harness.spec import (
    Commit0Spec,
    SimpleSpec,
    SWEBenchSpec,
    Spec,
    get_specs_from_dataset,
    make_spec,
)


def _make_commit0_spec(
    instance: RepoInstance,
    absolute: bool = True,
) -> Commit0Spec:
    repo_directory = "/testbed" if absolute else "testbed"
    return Commit0Spec(
        repo=instance["instance_id"],
        repo_directory=repo_directory,
        instance=instance,
        absolute=absolute,
    )


def _make_simple_spec(
    instance: SimpleInstance,
    absolute: bool = True,
) -> SimpleSpec:
    repo_directory = "/testbed" if absolute else "testbed"
    return SimpleSpec(
        repo="simple",
        repo_directory=repo_directory,
        instance=instance,
        absolute=absolute,
    )


def _make_swebench_spec(
    instance: RepoInstance,
    absolute: bool = True,
) -> SWEBenchSpec:
    repo_directory = "/testbed" if absolute else "testbed"
    return SWEBenchSpec(
        repo=instance["instance_id"],
        repo_directory=repo_directory,
        instance=instance,
        absolute=absolute,
    )


class TestMakeSpec:
    def test_make_spec_commit0(self, sample_repo_instance: RepoInstance) -> None:
        spec = make_spec(sample_repo_instance, "commit0", True)
        assert isinstance(spec, Commit0Spec)

    def test_make_spec_swebench(self, sample_repo_instance: RepoInstance) -> None:
        spec = make_spec(sample_repo_instance, "swebench", True)
        assert isinstance(spec, SWEBenchSpec)

    def test_make_spec_simple(self, sample_simple_instance: SimpleInstance) -> None:
        spec = make_spec(sample_simple_instance, "simple", True)
        assert isinstance(spec, SimpleSpec)

    def test_make_spec_unknown_raises(self, sample_repo_instance: RepoInstance) -> None:
        with pytest.raises(NotImplementedError, match="unknown is not supported"):
            make_spec(sample_repo_instance, "unknown", True)

    def test_make_spec_already_spec_returns_same(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        result = make_spec(spec, "commit0", True)  # type: ignore[arg-type]
        assert result is spec

    def test_make_spec_absolute_true_uses_testbed(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = make_spec(sample_repo_instance, "commit0", True)
        assert spec.repo_directory == "/testbed"

    def test_make_spec_absolute_false_uses_relative(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = make_spec(sample_repo_instance, "commit0", False)
        assert spec.repo_directory == "testbed"


class TestGetSpecsFromDataset:
    def test_converts_instances_to_specs(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        dataset = [sample_repo_instance]
        result = get_specs_from_dataset(dataset, "commit0", True)  # type: ignore[arg-type]
        assert len(result) == 1
        assert isinstance(result[0], Spec)

    def test_idempotent_with_specs(self, sample_repo_instance: RepoInstance) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        dataset: list[Spec] = [spec]
        result = get_specs_from_dataset(dataset, "commit0", True)
        assert result is dataset

    def test_empty_dataset_raises(self) -> None:
        empty: list[RepoInstance] = []
        with pytest.raises(IndexError):
            get_specs_from_dataset(empty, "commit0", True)  # type: ignore[arg-type]


class TestCommit0Spec:
    def test_make_repo_script_list_has_git_clone(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        script_list = spec.make_repo_script_list()
        joined = "\n".join(script_list)
        assert "git clone" in joined

    def test_make_repo_script_list_has_reference_commit(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        script_list = spec.make_repo_script_list()
        joined = "\n".join(script_list)
        assert sample_repo_instance.reference_commit in joined

    def test_make_eval_script_list_has_git_apply(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        script_list = spec.make_eval_script_list()
        joined = "\n".join(script_list)
        assert "git apply" in joined

    def test_setup_script_starts_with_shebang(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        assert spec.setup_script.startswith("#!/bin/bash")


class TestSimpleSpec:
    def test_make_repo_script_list_has_mkdir(
        self, sample_simple_instance: SimpleInstance
    ) -> None:
        spec = _make_simple_spec(sample_simple_instance)
        script_list = spec.make_repo_script_list()
        joined = "\n".join(script_list)
        assert "mkdir" in joined

    def test_make_eval_script_list_has_cat_patch(
        self, sample_simple_instance: SimpleInstance
    ) -> None:
        spec = _make_simple_spec(sample_simple_instance)
        script_list = spec.make_eval_script_list()
        joined = "\n".join(script_list)
        assert "cat /patch.diff" in joined

    def test_eval_script_has_pytest(
        self, sample_simple_instance: SimpleInstance
    ) -> None:
        spec = _make_simple_spec(sample_simple_instance)
        assert "pytest" in spec.eval_script


class TestSWEBenchSpec:
    def test_make_repo_script_list_has_clone(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_swebench_spec(sample_repo_instance)
        script_list = spec.make_repo_script_list()
        joined = "\n".join(script_list)
        assert "git clone" in joined

    def test_make_eval_script_list_with_install(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_swebench_spec(sample_repo_instance)
        script_list = spec.make_eval_script_list()
        joined = "\n".join(script_list)
        assert "pip install" in joined


class TestSpecProperties:
    def test_base_image_key_format(self, sample_repo_instance: RepoInstance) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        assert spec.base_image_key == "commit0.base.python3.12:latest"

    def test_repo_image_key_hash_consistency(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec1 = _make_commit0_spec(sample_repo_instance)
        spec2 = _make_commit0_spec(sample_repo_instance)
        key1 = spec1.repo_image_key
        key2 = spec2.repo_image_key
        assert key1 == key2
        assert key1 == key1.lower()

    def test_repo_image_key_different_scripts_differ(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec1 = _make_commit0_spec(sample_repo_instance)
        key1 = spec1.repo_image_key

        altered = sample_repo_instance.model_copy(
            update={"base_commit": "zzz999", "reference_commit": "yyy888"}
        )
        spec2 = _make_commit0_spec(altered)
        key2 = spec2.repo_image_key
        assert key1 != key2

    def test_get_container_name_without_run_id(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        name = spec.get_container_name()
        repo_part = spec.repo.split("/")[-1]
        assert name == f"commit0.eval.{repo_part}"

    def test_get_container_name_with_run_id(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        name = spec.get_container_name(run_id="Run123")
        repo_part = spec.repo.split("/")[-1]
        expected = f"commit0.eval.{repo_part}.Run123".lower()
        assert name == expected

    def test_platform_default(
        self, sample_repo_instance: RepoInstance, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("COMMIT0_BUILD_PLATFORMS", raising=False)
        spec = _make_commit0_spec(sample_repo_instance)
        assert spec.platform == "linux/amd64,linux/arm64"

    def test_platform_env_override(
        self, sample_repo_instance: RepoInstance, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COMMIT0_BUILD_PLATFORMS", "linux/amd64")
        spec = _make_commit0_spec(sample_repo_instance)
        assert spec.platform == "linux/amd64"

    def test_get_python_version_default(self) -> None:
        instance = RepoInstance(
            instance_id="test/no-py",
            repo="no-py-repo",
            base_commit="aaa",
            reference_commit="bbb",
            setup={},
            test={"test_cmd": "pytest"},
            src_dir="src",
        )
        spec = _make_commit0_spec(instance)
        assert spec._get_python_version() == "3.12"

    def test_get_python_version_from_setup(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        altered = sample_repo_instance.model_copy(update={"setup": {"python": "3.10"}})
        spec = _make_commit0_spec(altered)
        assert spec._get_python_version() == "3.10"

    def test_base_dockerfile_delegates(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        with patch(
            "commit0.harness.spec.get_dockerfile_base", return_value="FROM python:3.12"
        ) as mock_fn:
            result = spec.base_dockerfile
            mock_fn.assert_called_once_with("3.12")
            assert result == "FROM python:3.12"

    def test_repo_dockerfile_delegates(
        self, sample_repo_instance: RepoInstance
    ) -> None:
        spec = _make_commit0_spec(sample_repo_instance)
        with patch(
            "commit0.harness.spec.get_dockerfile_repo",
            return_value="FROM commit0.base:latest",
        ) as mock_fn:
            result = spec.repo_dockerfile
            mock_fn.assert_called_once()
            assert result == "FROM commit0.base:latest"
