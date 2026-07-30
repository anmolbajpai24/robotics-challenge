# Compatibility patches for gym-xarm 0.1.1 (written for gymnasium <=0.29, dormant upstream):
# restores the old render(camera_name=...) call style, restores renderer sizes from the
# mujoco model, and fixes is_success(), which upstream compares across two coordinate
# frames and is unsatisfiable as shipped. Import this BEFORE gym.make() in every script.
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gym_xarm.tasks.lift import Lift

_original_render = MujocoRenderer.render

def _render_with_camera(self, render_mode, camera_name=None):
    if camera_name is not None:
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    return _original_render(self, render_mode)

MujocoRenderer.render = _render_with_camera


_original_init = MujocoRenderer.__init__

def _init_with_model_size(self, model, data, *args, **kwargs):
    kwargs.setdefault("width", model.vis.global_.offwidth)
    kwargs.setdefault("height", model.vis.global_.offheight)
    _original_init(self, model, data, *args, **kwargs)

MujocoRenderer.__init__ = _init_with_model_size


def _is_success_relative(self):
    # success = cube >= 15 cm above its spawn height, all in table-relative coords
    return self.obj[2] >= (self._init_z - self.center_of_table[2]) + self._z_threshold

Lift.is_success = _is_success_relative
