# Day 21: resurrect the gym-xarm Push task. The installed push.py is a
# deprecated stub ("DEPRECATED: use only Lift for now") — the whole class is
# commented out, written for the old mujoco-py API (self.sim.*), and not
# registered. This file ports that commented implementation onto the modern
# Base class (which already provides eef/obj/obj_rot/... as properties) and
# registers gym_xarm/XarmPush-v0 with the same settings as lift's registration
# (max_episode_steps=300, obs_type="state"). The installed package is never
# edited.
#
# Deviations from the commented original, all forced by the modern Base:
# 1. is_success: the original compares self.obj with self.goal, but modern
#    Base.obj is TABLE-RELATIVE (base.py:175 subtracts center_of_table) while
#    self.goal is absolute world coords — the same cross-frame bug day 14
#    found in Lift. Distance is ~2 m forever, success unsatisfiable. Patched
#    like day14/xarm_compat.py: same threshold semantics (3D dist <= 0.05),
#    both points in the absolute frame.
# 2. action_space "xyzw" instead of the original "xyz": modern Base.step and
#    _apply_action hard-assert 4-dim actions and never apply the old 3-dim
#    action_padding, so a 3-dim space cannot step at all. The original padded
#    the gripper channel to a constant 1.0; our scripted oracle commands
#    gripper 1.0 (closed) throughout, preserving that behavior — and keeping
#    the action width at 4, same as day 14/16.
# 3. _reset_sim returns super()'s bool: the original returned None, which
#    would spin modern Base.reset's `while not did_reset_sim` loop forever.
# 4. _get_obs returns a flat vector (modern obs_type="state" contract), not
#    the original's dict; same values in the same order. Base.eef/obj are
#    already table-relative, so the original's `- self.center_of_table` is
#    NOT re-applied to them (only to goal) — no double subtraction.
#
# Also imports day14/xarm_compat for the renderer patches (camera_name call
# style + render sizes). Its Lift.is_success patch tags along harmlessly.
# Import this module BEFORE gym.make() in every script.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day14"))
import xarm_compat  # noqa: F401  (renderer patches, must precede gym.make)
import gymnasium as gym
from gymnasium.envs.registration import register
from gym_xarm.tasks import Base


class Push(Base):
    metadata = {
        **Base.metadata,
        "action_space": "xyzw",  # original said xyz; see deviation 2 above
        "episode_length": 50,
        "description": "Push a cube to a target location",
    }

    def __init__(self, **kwargs):
        self._act_magnitude = 0.0
        super().__init__("push", **kwargs)

    def _reset_sim(self):
        self._act_magnitude = 0.0
        return super()._reset_sim()  # deviation 3: must return the bool

    def is_success(self):
        # deviation 1: both points in the absolute frame, threshold unchanged
        cube_absolute = self.obj + self.center_of_table
        return np.linalg.norm(cube_absolute - self.goal) <= 0.05

    def get_reward(self):
        cube_absolute = self.obj + self.center_of_table
        distance = np.linalg.norm(cube_absolute - self.goal)
        penalty = self._act_magnitude**2
        return -(distance + 0.15 * penalty)

    def _get_obs(self):
        # eef/obj already table-relative in modern Base; goal stored absolute
        eef, obj = self.eef, self.obj
        goal = self.goal - self.center_of_table
        return np.concatenate(
            [
                eef,
                self.eef_velp,
                goal,
                obj,
                self.obj_rot,
                self.obj_velp,
                self.obj_velr,
                eef - goal,
                eef - obj,
                obj - goal,
                np.array(
                    [
                        np.linalg.norm(eef - goal),
                        np.linalg.norm(eef - obj),
                        np.linalg.norm(obj - goal),
                    ]
                ),
                self.gripper_angle,
            ],
            axis=0,
        )

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object: spawns 25 cm toward the robot from table center, +/- 8 cm xy
        object_pos = self.center_of_table - np.array([0.25, 0, 0.07])
        object_pos[0] += self.np_random.uniform(-0.08, 0.08)
        object_pos[1] += self.np_random.uniform(-0.08, 0.08)
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)

        # Goal: absolute world coords, varies +/- 10 cm in xy per episode
        self.goal = np.array([1.600, 0.200, 0.545])
        self.goal[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
        target_site_id = self._model_names.site_name2id["target0"]
        self.model.site_pos[target_site_id] = self.goal
        return self.goal

    def step(self, action):
        self._act_magnitude = np.linalg.norm(action[:3])
        return super().step(action)


if "gym_xarm/XarmPush-v0" not in gym.registry:
    register(
        id="gym_xarm/XarmPush-v0",
        entry_point=Push,  # class object directly; this module isn't a package
        max_episode_steps=300,
        kwargs={"obs_type": "state"},
    )
