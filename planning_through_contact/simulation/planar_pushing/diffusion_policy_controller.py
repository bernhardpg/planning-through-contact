from typing import List, Optional, Tuple
import logging

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pathlib
from pydrake.common.value import AbstractValue, Value
from pydrake.math import RigidTransform
from pydrake.systems.framework import (
    Context,
    LeafSystem,
)
from pydrake.systems.sensors import (
    PixelType,
    Image
)

import time as pytime


# Diffusion Policy imports
import torch
import dill
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset


from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import (
    HybridMpc,
    HybridMpcConfig,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)

logger = logging.getLogger(__name__)

# Set the print precision to 4 decimal places
np.set_printoptions(precision=4)

# TODO: hard code in temrination
# if object is within tolerance and pusher is near final position

class DiffusionPolicyController(LeafSystem):
    def __init__(
        self,
        checkpoint: str,
        diffusion_policy_path: str = "/home/adam/workspace/gcs-diffusion",
        initial_pusher_pose: PlanarPose=PlanarPose(
            x=0.5, y=0.25, theta=0.0
        ),
        target_slider_pose: PlanarPose=PlanarPose(
            x=0.5, y=0.0, theta=0.0
        ),
        freq: float = 10.0,
        delay=1.0,
    ):
        super().__init__()
        self._checkpoint = pathlib.Path(checkpoint)
        self._diffusion_policy_path = pathlib.Path(diffusion_policy_path)
        self._initial_pusher_pose = initial_pusher_pose
        self._target_slider_pose = target_slider_pose
        self._freq = freq
        self._dt = 1.0 / freq
        self._delay = delay
        self._load_policy_from_checkpoint(self._checkpoint)
        
        # get parameters
        self._obs_horizon = self._cfg.n_obs_steps
        self._action_steps = self._cfg.n_action_steps
        self._state_dim = self._cfg.shape_meta.obs.agent_pos.shape[0]
        self._action_dim = self._cfg.shape_meta.action.shape[0]
        self._target_dim = self._cfg.policy.target_dim
        self._num_image_channels = self._cfg.shape_meta.obs.image.shape[0]
        self._image_height = self._cfg.shape_meta.obs.image.shape[1]
        self._image_width = self._cfg.shape_meta.obs.image.shape[2]
        self._B = 1 # batch size is 1

        # indexing parameters for action predictions
        self._start = self._obs_horizon - 1
        self._end = self._start + self._action_steps
        # Hack to ensure backward compatibility with version 2 checkpoints
        # Version 2 checkpoints did not used shifted actions
        if 'push_tee_v2' in checkpoint:
            print("Using version 2 slicing for action predictions")
            self._start = self._obs_horizon
            self._end = self._start + self._action_steps
        
        # observation histories
        self._pusher_pose_deque = deque(
            [self._initial_pusher_pose.vector()
            for _ in range(self._obs_horizon)], 
            maxlen=self._obs_horizon
        )
        self._image_deque = deque([], maxlen=self._obs_horizon)

        # variables for DoCalcOutput
        self._actions = deque([], maxlen=self._action_steps)
        self._current_action = np.array([
            self._initial_pusher_pose.x,
            self._initial_pusher_pose.y,
        ])

        # Input port for pusher pose
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_measured",
            AbstractValue.Make(RigidTransform()),
        )
        self.camera_port = self.DeclareAbstractInputPort(
            "camera",
            Value[Image[PixelType.kRgba8U]].Make(
                Image[PixelType.kRgba8U](self._image_width, self._image_height)
            ),
        )
        
        self.output = self.DeclareVectorOutputPort(
            "planar_position_command", 
            2, 
            self.DoCalcOutput
        )
    
    def _load_policy_from_checkpoint(self, checkpoint: str):
        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        self._cfg = payload['cfg']
        cls = hydra.utils.get_class(self._cfg._target_)
        workspace: BaseWorkspace
        workspace = cls(self._cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get normalizer: this might be expensive for larger datasets
        self._cfg.task.dataset.zarr_path = self._diffusion_policy_path.joinpath(
            self._cfg.task.dataset.zarr_path
        )
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(self._cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        self._normalizer = dataset.get_normalizer()

        # get policy from workspace
        self._policy = workspace.model
        self._policy.set_normalizer(self._normalizer)
        if self._cfg.training.use_ema:
            self._policy = workspace.ema_model
            self._policy.set_normalizer(self._normalizer)
        
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._policy.to(self._device)
        self._policy.eval()
    
    def DoCalcOutput(self, context: Context, output):
        time = context.get_time()
        
        # Continually update ports until delay is over
        if time < self._delay:
            self._update_history(context)
            output.set_value(self._current_action)
            return
        
        if len(self._pusher_pose_deque) < self._obs_horizon or len(self._image_deque) < self._obs_horizon:
            self._update_history(context)
            output.set_value(self._current_action)
            return


        # Update observation history
        self._update_history(context)

        obs_dict = self._deque_to_dict(
            self._pusher_pose_deque,
            self._image_deque,
            self._target_slider_pose.vector()
        )
        
        # Actions available: use next action
        if len(self._actions) == 0:
            # print(f"Time: {time:.3f}, state_deque: {self._pusher_pose_deque}")
            # for img in self._image_deque:
            #     plt.imshow(img)
            #     plt.show()
            start_time = pytime.time()
            with torch.no_grad():
                action_prediction = self._policy.predict_action(obs_dict)['action_pred'][0]
          
            actions = action_prediction[self._start:self._end]
            for action in actions:
                self._actions.append(action.cpu().numpy())
            # print(f"[TIME: {time:.3f}] Computed new actions in {pytime.time() - start_time:.3f}s")

            # DEBUG: dummy actions (move pusher in positive x direction)
            # new_desired_pose = self._pusher_pose_deque[-1].copy()
            # for i in range(self._action_steps):
            #     self._actions.append(np.array(
            #         [new_desired_pose[0], new_desired_pose[1]]
            #     ))
            #     new_desired_pose[0] += 0.01 # move freq*0.01m/s

        # get next action and increment next update time
        assert len(self._actions) > 0
        prev_action = self._current_action
        self._current_action = self._actions.popleft()
        output.set_value(self._current_action)

        # debug print statements
        delta = np.linalg.norm(self._current_action - prev_action)
        # print(f"Time: {time:.3f}, action delta: {delta}")
        # print(f"Time: {time:.3f}, action: {self._current_action}")
        
    def reset(self, reset_position: np.ndarray):
        self._current_action = reset_position
        self._actions.clear()
        self._pusher_pose_deque.clear()
        self._image_deque.clear()
    
    def _deque_to_dict(self, 
                      obs_deque: deque, 
                      img_deque: deque, 
                      target: np.ndarray
                    ):      
        state_tensor = torch.cat(
            [torch.from_numpy(obs) for obs in obs_deque], 
            dim=0
        ).reshape(self._B, self._obs_horizon, self._state_dim)
        img_tensor = torch.cat(
            [torch.from_numpy(np.moveaxis(img,-1,-3) / 255.0) for img in img_deque], 
            dim=0
        ).reshape(self._B, self._obs_horizon, self._num_image_channels, self._image_width, self._image_height)
        target_tensor = torch.from_numpy(target).reshape(1, self._target_dim) # 1, D_t
        return {'obs': {
                    'image': img_tensor.to(self._device), # 1, T_obs, C, H, W
                    'agent_pos': state_tensor.to(self._device), # 1, T_obs, D_x
                },
                'target': target_tensor.to(self._device), # 1, D_t
        }

    def _update_history(self, context):
        """ Update state and image observation history """
        pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
        image = self.camera_port.Eval(context)
        pusher_planer_pose = PlanarPose.from_pose(pusher_pose).vector()
        self._pusher_pose_deque.append(pusher_planer_pose)
        self._image_deque.append(image.data[:,:,:-1])