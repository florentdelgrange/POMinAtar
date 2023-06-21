# Adapted from https://github.com/qlan3/gym-games
import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register

try:
    import seaborn as sns
except:
    import logging
    logging.warning("Cannot import seaborn."
        "Will not be able to train from pixel observations.")

from minatar import Environment


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "array", "rgb_array"]}

    def __init__(self, game, render_mode=None, display_time=50,
                use_minimal_action_set=False, **kwargs):
        self.render_mode = render_mode
        self.display_time = display_time

        self.game = Environment(env_name=game, **kwargs)

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            0, 1, shape=self.game.state_shape(), dtype=np.uint8
        )

    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        if self.render_mode == "human":
            self.render()
        return self.game.observation, reward, done, False, {}

    def seed(self, seed=None):
        self.game.seed(seed)

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        if self.render_mode == "human":
            self.render()
        return self.game.observation, {}

    def render(self, mode: str = None):
        if self.render_mode is None and mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if mode is None:
            mode = self.render_mode
        if mode == "array":
            return self.game.state()
        elif mode == "human":
            self.game.display_state(self.display_time)
        elif mode == "rgb_array": # use the same color palette of Environment.display_state
            state = self.game.state()
            n_channels = state.shape[-1]
            cmap = sns.color_palette("cubehelix", n_channels)
            cmap.insert(0, (0,0,0))
            numerical_state = np.amax(
                state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
            rgb_array = np.stack(cmap)[numerical_state]
            return rgb_array

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0


def register_envs():
    for game in ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]:
        name = game.title().replace('_', '').capitalize()
        kwargs = dict(game=game, use_minimal_action_set=False)
        if game == "breakout":
            kwargs['no_ball'] = False
        if game == "seaquest":
            kwargs['oxygen_noise'] = False
        if game == "space_invaders":
            kwargs['noise'] = False

        def _register(params):
            if params:
                kwarg = params[0]
                kwargs[kwarg] = False

                register(
                    id="MinAtar/{}-v{}".format(name, 1 if kwargs['use_minimal_action_set'] else 0),
                    entry_point="minatar.gym:BaseEnv",
                    kwargs=kwargs,
                )
                _register(params[1:])

                kwarg[kwarg] = True

                if kwarg == 'use_minimal_action_set':
                    kwarg_name = ''
                else:
                    kwarg_name = ''.join(x.capitalize() or '_' for x in kwarg.split('_'))

                register(
                    id="MinAtar/{}{}-v{}".format(name, kwarg_name, 1 if kwargs['use_minimal_action_set'] else 0),
                    entry_point="minatar.gym:BaseEnv",
                    kwargs=kwargs,
                )
                _register(params[1:])

        params = list(kwargs.keys())
        params.remove('game')
        _register(params=list(kwargs.keys()))