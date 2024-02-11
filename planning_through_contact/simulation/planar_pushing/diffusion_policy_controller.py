from typing import List, Optional, Tuple
import logging

import numpy as np
import numpy.typing as npt
from collections import deque
import pathlib
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.systems.framework import (
    AbstractStateIndex,
    Context,
    ContinuousState,
    LeafSystem,
    AbstractStateIndex,
    State,
)
from pydrake.systems.sensors import (
    PixelType,
    Image
)

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

class DiffusionPolicyController(LeafSystem):
    def __init__(
        self,
        checkpoint: str,
        initial_pusher_pose: PlanarPose=PlanarPose(
            x=0.575, y=0.25, theta=0.0
        ),
        freq: float = 10.0,
        delay=1.0
    ):
        super().__init__()
        self._checkpoint = pathlib.Path(checkpoint)
        self._initial_pusher_pose = initial_pusher_pose
        self._freq = freq
        self._dt = 1.0 / freq
        # TODO: uncomment this once connections are verified
        # self._load_policy_from_checkpoint(self._checkpoint)

        # get parameters
        self._obs_horizon = self._cfg.policy.n_obs_steps
        self._action_steps = self._cfg.n_action_steps
        self._B = 1 # batch size is 1
        
        # observation histories
        self._pusher_pose_deque = deque(
            [self._initial_pusher_pose for _ in range(self._obs_horizon)], 
            maxlen=2
        )
        self._image_deque = deque([], maxlen=2)

        # variables for DoCalcOutput
        self._actions = deque([], maxlen=self._action_steps)
        self._current_action: PlanarPose = self._initial_pusher_pose
        self._next_update_time = self._delay # tracks when to update actions

        # Input port for pusher pose
        self.pusher_pose_measured = self.DeclareAbstractInputPort(
            "pusher_pose_estimated",
            AbstractValue.Make(RigidTransform()),
        )
        self.camera_port = self.DeclareAbstractInputPort(
            "camera",
            AbstractValue.Make(Image[PixelType.kRgba8U]),
        )
        
        self.output = self.DeclareVectorOutputPort(
            "translation", 
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
        if time < self._next_update_time:
            output.set_value(self._current_action.pos())
            return

        # Read input ports
        pusher_pose: RigidTransform = self.pusher_pose_measured.Eval(context)  # type: ignore
        pusher_planar_pose = PlanarPose.from_pose(pusher_pose)
        image = self.camera_port.Eval(context) # TODO: convert image to numpy array/tensor

        # Update observation history
        self._pusher_pose_deque.append(pusher_planar_pose)
        self._image_deque.append(image)

        # Actions available: use next action
        if len(self._actions) == 0:
            # TODO: actual diffusion policy code here
            new_desired_pose = pusher_planar_pose
            for i in range(self._action_steps):
                self._actions.append(new_desired_pose)
                new_desired_pose.x += 0.01 # move freq*0.01m/s

        # get next action and increment next update time
        assert len(self._actions) > 0
        self._current_action = self._actions.popleft()
        output.set_value(self._current_action.pos())
        self._next_update_time += self._dt