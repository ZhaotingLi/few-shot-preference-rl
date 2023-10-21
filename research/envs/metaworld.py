"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gymnasium as gym
# import gym
import metaworld
import numpy as np


class SawyerEnv(gym.Env):
    def __init__(self, env_name, seed=True):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        print("env_name: ", env_name)
        self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.unwrapped.seed(0)  # Seed it at zero for now.

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._max_episode_steps = self._env.max_path_length
        self.max_episode_steps = self._env.max_path_length

    def seed(self, seed=None):
        # super().seed(seed=seed)
        if self._seed:
            self._env.unwrapped.seed(0)

    def evaluate_state(self, state, action):
        return self._env.evaluate_state(state, action)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        if self._episode_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        ## Add the underlying state to the info
        # metaworld doesn't have a sim object now, should find other ways to get the state
        # state = self._env.sim.get_state()
        # info["state"] = np.concatenate((state.qpos, state.qvel), axis=0)
        state = self._env.get_env_state()
        info["state"] = np.concatenate((state[0], state[1]), axis=0)
        return obs.astype(np.float32), reward, done, info

    def set_state(self, state):
        qpos, qvel = state[: self._env.model.nq], state[self._env.model.nq :]
        # self._env.set_state(qpos, qvel)
        self._env.set_env_state((qpos, qvel))

    def reset(self, **kwargs):
        self._episode_steps = 0
        # return self._env.reset(**kwargs).astype(np.float32)  # changed, as the new gymnasium version returns a tuple (obs, info)
        return self._env.reset(**kwargs)[0].astype(np.float32)

    def render_(self, mode="rgb_array", width=640, height=480):
        assert mode == "rgb_array", "Only RGB array is supported"
        # stack multiple views
        # the render function is defined here: https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L404 
        self._env.render_mode = mode
        self._env.camera_name = "corner3" #"corner2"  #"leftcam"  #"corner"
        view_1 = self._env.render()
        self._env.camera_name = "topview"
        view_2 = self._env.render()


        # view_1 = self._env.render(offscreen=True, camera_name="corner", resolution=(width, height))
        # view_2 = self._env.render(offscreen=True, camera_name="topview", resolution=(width, height))
        return np.concatenate((view_1, view_2), axis=0)
