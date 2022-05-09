"""
AWR + SAC from demo experiment
"""

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import StackObservationEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.torch.networks.mlp import MlpQfWithObsProcessor, MlpVfWithObsProcessor
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader

from rlkit.launchers.experiments.awac.finetune_rl import get_normalization

import torch
import numpy as np

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
import pickle

# from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.envs.make_env import make

from rlkit.torch.sac.policies.gaussian_policy import GaussianPolicyAdapter
from rlkit.torch.sac.iql_trainer import IQLTrainer

import random
from rlkit.torch.sac.policies.gaussian_policy import ImageObsProcessor


variant = dict(
    algo_kwargs=dict(
        start_epoch=-1000, # offline epochs
        num_epochs=0, # online epochs
        batch_size=256,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
    ),
    max_path_length=1000,
    replay_buffer_size=int(2E6),
    layer_size=256,
    policy_class=GaussianPolicyAdapter,
    policy_kwargs=dict(
        hidden_sizes=[256, 256, ],
    ),
    qf_kwargs=dict(
        hidden_sizes=[256, 256, ],
    ),
    img_obs_processor_kwargs=dict(
        use_r3m=True,
        freeze=True
    ),
    algorithm="SAC",
    version="normal",
    collection_mode='batch',
    trainer_class=IQLTrainer,
    trainer_kwargs=dict(
        discount=0.99,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        soft_target_tau=0.005,

        policy_weight_decay=0,
        q_weight_decay=0,

        reward_transform_kwargs=None,
        terminal_transform_kwargs=None,

        beta=1.0 / 3,
        quantile=0.8,
        clip_score=100,
    ),
    path_loader_class=HDF5PathLoader,
    path_loader_kwargs=dict(),
    add_env_demos=False,
    add_env_offpolicy_data=False,

    load_demos=True,
    normalize_env=False,
    env_id='drawer-open-v2',
    normalize_rewards_by_return_range=True,

    seed=random.randint(0, 100000),
)


def experiment(variant):
    normalize_env = variant.get('normalize_env', True)
    env_id = variant.get('env_id', None)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', {})

    expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    eval_env = make(env_id, env_class, env_kwargs, normalize_env)

    seed = int(variant["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env.seed(seed)
    expl_env.seed(seed)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    qf_kwargs = variant.get("qf_kwargs", {})

    policy_img_obs_processor = ImageObsProcessor(**variant['img_obs_processor_kwargs'])
    qf_img_obs_processor = ImageObsProcessor(**variant['img_obs_processor_kwargs'])
    target_img_obs_processor = ImageObsProcessor(**variant['img_obs_processor_kwargs'])

    qf1 = MlpQfWithObsProcessor(
        obs_processor=qf_img_obs_processor,
        input_size=obs_dim + action_dim + 512,
        output_size=1,
        **qf_kwargs
    )
    qf2 = MlpQfWithObsProcessor(
        obs_processor=qf_img_obs_processor,
        input_size=obs_dim + action_dim + 512,
        output_size=1,
        **qf_kwargs
    )
    target_qf1 = MlpQfWithObsProcessor(
        obs_processor=target_img_obs_processor,
        input_size=obs_dim + action_dim + 512,
        output_size=1,
        **qf_kwargs
    )
    target_qf2 = MlpQfWithObsProcessor(
        obs_processor=target_img_obs_processor,
        input_size=obs_dim + action_dim + 512,
        output_size=1,
        **qf_kwargs
    )

    vf_kwargs = variant.get("vf_kwargs", dict(hidden_sizes=[256, 256, ],))
    vf = MlpVfWithObsProcessor(
        input_size=obs_dim + 512,
        output_size=1,
        **vf_kwargs
    )

    policy_class = variant.get("policy_class", TanhGaussianPolicy)

    policy_kwargs = variant['policy_kwargs']
    policy = policy_class(
        obs_processor=policy_img_obs_processor,
        obs_processor_output_dim=obs_dim + 512,
        action_dim=action_dim,
        **policy_kwargs,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpsilonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **replay_buffer_kwargs,
    )
    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    trainer_class = variant.get("trainer_class", AWACTrainer)
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        **variant['trainer_kwargs']
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)

    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('save_initial_buffers', False):
        buffers = dict(
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
        )
        buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
        pickle.dump(buffers, open(buffer_path, "wb"))

    algorithm.train()

experiment(variant)
