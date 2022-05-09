import gym
import numpy as np


class ImgObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.img_res = (64, 64)
        low_1 = self.env.observation_space.low[:4]
        low_2 = self.env.observation_space.low[7:11]
        high_1 = self.env.observation_space.high[:4]
        high_2 = self.env.observation_space.high[7:11]
        low = np.hstack((low_1, low_2))
        high = np.hstack((high_1, high_2))
        self.observation_space = gym.spaces.Box(low, high)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        img = self.env.render(offscreen=True, camera_name="behindGripper", resolution=self.img_res)
        next_state = self._remove_non_proprio(next_state)
        next_obs = {'state': next_state, 'img': img}
        return next_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._remove_non_proprio(obs)
        img = self.env.render(offscreen=True, camera_name="behindGripper", resolution=self.img_res)
        obs = {'state': obs, 'img': img}
        return obs

    def _remove_non_proprio(self, obs):
        hand_and_gripper_1 = obs[..., :4]
        hand_and_gripper_2 = obs[..., 7:11]
        return np.hstack((hand_and_gripper_1, hand_and_gripper_2))
