import numpy as np

from most.data.datasets.robocasa365_dataset import (
    robocasa_action_exporter_to_policy,
)


def _construct_exporter_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "action.base_motion": action[0:4],
        "action.control_mode": action[4:5],
        "action.end_effector_position": action[5:8],
        "action.end_effector_rotation": action[8:11],
        "action.gripper_close": action[11:12],
    }


def test_action_reorder_known_indices() -> None:
    exporter = np.arange(12, dtype=np.float32)
    policy = robocasa_action_exporter_to_policy(exporter)
    np.testing.assert_array_equal(
        policy,
        np.asarray([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], dtype=np.float32),
    )


def test_action_reorder_matches_robocasa_convert_action() -> None:
    from robocasa.utils.env_utils import convert_action

    exporter = np.asarray(
        [0.11, 0.12, 0.13, 0.14, 1.0, 0.21, 0.22, 0.23, 0.31, 0.32, 0.33, 0.8],
        dtype=np.float32,
    )
    expected = _construct_exporter_dict(exporter)
    actual = convert_action(robocasa_action_exporter_to_policy(exporter))

    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(expected[key]))
