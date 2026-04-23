
import numpy as np
import torch
import os
import time

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import gym
import hydra


def make_env(cfg):
    """Create Gym environment (Pendulum-v1)."""
    env = gym.make(cfg.env)

    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # ---- ENV ---- #
        self.env = make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device
        )

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

        self.step = 0

    # ---------------- EVALUATION ---------------- #
    def evaluate(self):
        average_episode_reward = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))

            episode_reward = 0
            episode_step = 0

            while episode_step < self.env._max_episode_steps:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                obs, reward, terminated, truncated, _ = self.env.step(action)

                # ---- CUSTOM REWARD ---- #
                theta = np.arctan2(obs[1], obs[0])
                theta_target = self.cfg.theta_target

                angle_error = theta - theta_target
                angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

                reward = np.cos(angle_error) - 0.1 * (obs[2] ** 2) - 0.001 * (action ** 2)

                self.video_recorder.record(self.env)

                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    # ---------------- TRAIN LOOP ---------------- #
    def run(self):
        episode, episode_reward = 0, 0
        start_time = time.time()

        obs, _ = self.env.reset()
        self.agent.reset()
        episode_step = 0

        while self.step < self.cfg.num_train_steps:

            # ---- EVALUATION ---- #
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

            # ---- ACTION ---- #
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # ---- UPDATE ---- #
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            # ---- ENV STEP ---- #
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            # ---- CUSTOM REWARD ---- #
            theta = np.arctan2(next_obs[1], next_obs[0])
            theta_target = self.cfg.theta_target

            angle_error = theta - theta_target
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

            reward = np.cos(angle_error) - 0.1 * (next_obs[2] ** 2) - 0.001 * (action ** 2)

            # ---- NON-TERMINATING ---- #
            done = 0.0
            done_no_max = 0.0

            episode_reward += reward

            # ---- STORE ---- #
            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

            # ---- EPISODE RESET (manual) ---- #
            if episode_step >= self.env._max_episode_steps:
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/episode', episode, self.step)

                obs, _ = self.env.reset()
                self.agent.reset()

                episode_reward = 0
                episode_step = 0
                episode += 1


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()