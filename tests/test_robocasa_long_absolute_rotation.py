import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from most.data.datasets.robocasa_long_dataset import (
    absolute_rotation_targets,
    normalize_rotation_action_mode,
    validate_rotation_reference_quaternion_xyzw,
)


IDENTITY_QUATERNION_XYZW = np.asarray([0.0, 0.0, 0.0, 1.0])


def _states_with_x_rotations(angles: list[float]) -> np.ndarray:
    state = np.zeros((len(angles), 16), dtype=np.float32)
    state[:, 10:14] = Rotation.from_rotvec(
        np.asarray([[angle, 0.0, 0.0] for angle in angles])
    ).as_quat()
    return state


def test_absolute_rotation_target_uses_next_state() -> None:
    state = _states_with_x_rotations([0.0, 0.1, 0.2, 0.3, 0.4])
    target = absolute_rotation_targets(
        state,
        frame_index=0,
        action_length=3,
        reference_quaternion_xyzw=IDENTITY_QUATERNION_XYZW,
    )
    np.testing.assert_allclose(target[:, 0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(target[:, 1:], 0.0, atol=1e-6)


def test_absolute_rotation_target_clamps_episode_tail() -> None:
    state = _states_with_x_rotations([0.0, 0.1, 0.2, 0.3, 0.4])
    target = absolute_rotation_targets(
        state,
        frame_index=3,
        action_length=3,
        reference_quaternion_xyzw=IDENTITY_QUATERNION_XYZW,
    )
    np.testing.assert_allclose(target[:, 0], [0.4, 0.4, 0.4], atol=1e-6)


def test_reference_centering_is_continuous_across_world_pi_branch() -> None:
    """World +pi/-pi is local -epsilon/+epsilon around the reference."""
    angles = np.asarray([np.pi, np.pi - 0.02, np.pi + 0.02])
    state = np.zeros((len(angles), 16), dtype=np.float32)
    state[:, 10:14] = Rotation.from_rotvec(
        np.column_stack((np.zeros_like(angles), np.zeros_like(angles), angles))
    ).as_quat()
    reference = Rotation.from_rotvec([0.0, 0.0, np.pi]).as_quat()
    target = absolute_rotation_targets(
        state,
        frame_index=0,
        action_length=2,
        reference_quaternion_xyzw=reference,
    )
    np.testing.assert_allclose(target[:, :2], 0.0, atol=1e-6)
    np.testing.assert_allclose(target[:, 2], [-0.02, 0.02], atol=1e-6)
    assert np.linalg.norm(target[1] - target[0]) < 0.05


@pytest.mark.parametrize(
    "reference",
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, np.nan, 0.0, 1.0]),
)
def test_rotation_reference_is_strictly_validated(reference: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_rotation_reference_quaternion_xyzw(reference)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("delta", "delta"), ("relative", "delta"), ("abs", "absolute_rotvec")],
)
def test_rotation_action_mode_aliases(value: str, expected: str) -> None:
    assert normalize_rotation_action_mode(value) == expected


def _load_eval_entry():
    pytest.importorskip("robocasa")
    path = Path(__file__).resolve().parents[2] / "eval_robocasa_long" / "entry.py"
    spec = importlib.util.spec_from_file_location("robocasa_long_eval_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeOSC:
    input_type = "delta"
    input_ref_frame = "base"
    input_min = np.full(6, -1.0)
    input_max = np.full(6, 1.0)
    output_min = np.asarray([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5])
    output_max = np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])

    def __init__(self) -> None:
        self.current_site = np.eye(3)
        self.goal_ori = np.eye(3)

    def goal_origin_to_eef_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.current_site
        return pose


def test_absolute_rotation_closed_loop_inverse_osc_scale() -> None:
    entry = _load_eval_entry()
    observation = {"robot0_base_to_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0])}
    command = entry.absolute_rotvec_to_controller_input(
        np.asarray([0.1, 0.0, 0.0]),
        observation,
        _FakeOSC(),
        base_mode=False,
        rotation_reference_quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(command, [0.2, 0.0, 0.0], atol=1e-6)


def test_absolute_rotation_base_mode_uses_previous_desired_goal() -> None:
    entry = _load_eval_entry()
    observation = {"robot0_base_to_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0])}
    controller = _FakeOSC()
    controller.goal_ori = Rotation.from_rotvec([0.05, 0.0, 0.0]).as_matrix()
    command = entry.absolute_rotvec_to_controller_input(
        np.asarray([0.1, 0.0, 0.0]),
        observation,
        controller,
        base_mode=True,
        rotation_reference_quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(command, [0.1, 0.0, 0.0], atol=1e-6)


def test_delta_mode_remains_byte_for_byte_compatible() -> None:
    entry = _load_eval_entry()
    policy = np.asarray(
        [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.2, 0.7, 0.8, 0.9, 1.0, 0.8],
        dtype=np.float32,
    )
    actual = entry.policy_action_to_robosuite_action(policy)
    expected = policy.copy()
    expected[6] = -1.0
    expected[11] = 1.0
    np.testing.assert_array_equal(actual, expected)


def test_absolute_rotation_is_offset_from_exported_reference() -> None:
    entry = _load_eval_entry()
    observation = {"robot0_base_to_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0])}
    reference = Rotation.from_rotvec([0.1, 0.0, 0.0]).as_quat()
    command = entry.absolute_rotvec_to_controller_input(
        np.zeros(3),
        observation,
        _FakeOSC(),
        base_mode=False,
        rotation_reference_quaternion_xyzw=reference,
    )
    np.testing.assert_allclose(command, [0.2, 0.0, 0.0], atol=1e-6)


def test_absolute_rotation_preserves_body_to_site_offset() -> None:
    entry = _load_eval_entry()
    current_body = Rotation.from_rotvec([0.0, 0.0, 0.3])
    body_to_site = Rotation.from_rotvec([0.0, 0.2, 0.0])
    controller = _FakeOSC()
    controller.current_site = (current_body * body_to_site).as_matrix()
    observation = {"robot0_base_to_eef_quat": current_body.as_quat()}
    goal_body = Rotation.from_rotvec([0.1, 0.0, 0.0])
    command = entry.absolute_rotvec_to_controller_input(
        np.zeros(3),
        observation,
        controller,
        base_mode=False,
        rotation_reference_quaternion_xyzw=goal_body.as_quat(),
    )
    physical_error = (goal_body * current_body.inv()).as_rotvec()
    # The fake OSC maps +/-0.5 rad to +/-1 controller input.
    np.testing.assert_allclose(command, physical_error / 0.5, atol=1e-6)


def _write_model_config(path: Path, payload: dict) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return path


def test_eval_contract_rejects_forgotten_absolute_mode(tmp_path: Path) -> None:
    entry = _load_eval_entry()
    model = _write_model_config(
        tmp_path / "model",
        {
            "robocasa_long_rotation_action_mode": "absolute_rotvec",
            "robocasa_long_rotation_representation": (
                "base_relative_eef_local_rotvec_radians_about_dataset_reference_"
                "from_state_t_plus_1"
            ),
            "robocasa_long_robot_type": "robocasa_long_absrot",
            "robocasa_long_rotation_reference_quaternion_xyzw": [0, 0, 0, 1],
        },
    )
    with pytest.raises(ValueError, match="does not match the exported model"):
        entry.validate_eval_model_contract(str(model), "delta", "robocasa365")


def test_eval_contract_requires_reference_metadata(tmp_path: Path) -> None:
    entry = _load_eval_entry()
    model = _write_model_config(
        tmp_path / "model",
        {
            "robocasa_long_rotation_action_mode": "absolute_rotvec",
            "robocasa_long_rotation_representation": (
                "base_relative_eef_local_rotvec_radians_about_dataset_reference_"
                "from_state_t_plus_1"
            ),
            "robocasa_long_robot_type": "robocasa_long_absrot",
        },
    )
    with pytest.raises(ValueError, match="missing robocasa_long_rotation_reference"):
        entry.validate_eval_model_contract(
            str(model), "absolute_rotvec", "robocasa_long_absrot"
        )


def test_exported_hf_directory_beats_parent_training_config(tmp_path: Path) -> None:
    entry = _load_eval_entry()
    (tmp_path / "config.py").write_text(
        "data = {'params': {'train_datasets': {'processor_path': 'wrong'}}}\n",
        encoding="utf-8",
    )
    model = _write_model_config(
        tmp_path / "export" / "model",
        {"memory_enabled": True, "memory_image_size": 192},
    )
    processor_path, enabled, image_size, _tokens, _stride = (
        entry._resolve_eval_model_info(str(model))
    )
    assert processor_path == str(model)
    assert enabled is True
    assert image_size == 192
