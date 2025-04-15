from typing import List

import lcm
import numpy as np
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.types import ArticulationAction

from grutopia.core.robot.controller import BaseController
from grutopia.core.robot.robot import BaseRobot
from grutopia_extension.configs.controllers import GR1TeleOpControllerCfg
from grutopia_extension.controllers.lcmtypes.teleop import action, joints

from grutopia_extension.controllers import GR1T2TeleOpHandler


@BaseController.register('GR1TeleOpController')
class GR1TeleOpController(BaseController):
    """TeleOp controller for GR1."""

    def __init__(self, config: GR1TeleOpControllerCfg, robot: BaseRobot, scene: Scene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)

        urdf_path = '../../assets/robots/gr1/urdf/robot.urdf'
        retargeting_config_path = '../../assets/robots/gr1/inspire_hand/inspire_hand.yml'

        logging.info(f'args: urdf_path:{ urdf_path}, retargeting_config_path:{retargeting_config_path}')
        teleop = GR1T2TeleOpHandler(urdf_path, retargeting_config_path)
        logging.info('waiting for teleop action...')
        # while True:
        #     teleop.lc.handle()

        self.joint_names = config.joint_names
        self.joint_subset = ArticulationSubset(self.robot.isaac_robot, self.joint_names)

        self.teleop_joints = joints()

        # self.lc = lcm.LCM()
        # self.lc.subscribe('teleop_joints', self.teleop_joints_handler)

    def forward(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        head_pose: np.ndarray,
        left_hand_mat: np.ndarray,
        right_hand_mat: np.ndarray,
    ) -> ArticulationAction:
        teleop_action = action()
        teleop_action.left_wrist_mat = left_pose.tolist()
        teleop_action.right_wrist_mat = right_pose.tolist()
        teleop_action.head_mat = head_pose.tolist()
        teleop_action.left_hand_mat = left_hand_mat.tolist()
        teleop_action.right_hand_mat = right_hand_mat.tolist()

        # Send action and wait for joint positions.
        self.teleop_joints = teleop.action_to_control(_,teleop_action)
        self.last_joint_positions = np.array(self.teleop_joints.joint_positions)


        # self.lc.publish('teleop_action', teleop_action.encode())
        # self.lc.handle()

        return self.joint_subset.make_articulation_action(
            joint_positions=self.last_joint_positions, joint_velocities=None
        )

    def teleop_joints_handler(self, channel, data):
        teleop_joints = joints.decode(data)
        self.last_joint_positions = np.array(teleop_joints.joint_positions)

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (List | np.ndarray): 3-element 1d array containing:
              0. left wrist pose (np.ndarray)
              1. right wrist pose (np.ndarray)
              2. head pose (np.ndarray)
              3. left hand matrix (np.ndarray)
              4. right hand matrix (np.ndarray)

        Returns:
            ArticulationAction: joint signals to apply.
        """
        assert len(action) == 5, 'action must contain 5 elements'

        return self.forward(
            left_pose=action[0],
            right_pose=action[1],
            head_pose=action[2],
            left_hand_mat=action[3],
            right_hand_mat=action[4],
        )
