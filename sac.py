from gym.envs.mujoco import HalfCheetahEnv
import pickle
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import mt_rollout
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from pyvirtualdisplay import Display
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

import numpy as np
import metaworld
import rlkit.torch.pytorch_util as ptu
import sys
from pympler import asizeof
import os

def experiment(variant):
    mt1 = metaworld.MT1('drawer-open-v2') # Construct the benchmark, sampling tasks

    expl_env = mt1.train_classes['drawer-open-v2']()  # Create an environment with task `pick_place`
    eval_env = mt1.train_classes['drawer-open-v2']()
    #expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    #eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        rollout_fn = mt_rollout,
        benchmark = mt1,
        img_res = variant['img_res']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        env_info_sizes={'img':(*variant['img_res'], 3), 'next_img':(*variant['img_res'], 3)}
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
    output_dir = "exps/%s/" % variant['exp_name']
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir + "img.npy", replay_buffer._env_infos['img'])
    np.save(output_dir + "next_img.npy", replay_buffer._env_infos['next_img'])
    np.save(output_dir + "observations.npy", replay_buffer._observations)
    np.save(output_dir + "next_obs.npy", replay_buffer._next_obs)
    np.save(output_dir + "actions.npy", replay_buffer._actions)
    np.save(output_dir + "rewards.npy", replay_buffer._rewards)
    np.save(output_dir + "terminals.npy", replay_buffer._terminals)
    #print(asizeof.asizeof(replay_buffer))
    #with open('tests/test_1m.pkl', 'wb') as f:
    #    pickle.dump(replay_buffer, f)



if __name__ == "__main__":
    # noinspection PyTypeChecker
    #ptu.set_gpu_mode(True)
    display = Display(visible=0, size=(1400, 900))
    display.start()

    exp_name = "collect_1m_1000_epochs"

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e6),
        img_res = (64, 64),
        exp_name=exp_name,
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=500,
            batch_size=256 
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger(exp_name, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
