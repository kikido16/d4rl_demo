from tkinter import N
from d3rlpy.datasets import get_minari
from d3rlpy import algos
from d3rlpy import metrics
from d3rlpy import logging
from d3rlpy import load_learnable
from d3rlpy.models.encoders import VectorEncoderFactory
import numpy as np
import gym
import gymnasium
import gymnasium_robotics
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run offline rl with d3rlpy and Minari')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'test_env', 'evaluate'],
        help='Offline training or evaluation (default: train)'
    )
    parser.add_argument(
        '--choose_algorithm',
        type=str,
        default='iql',
        choices=['cql', 'bc', 'iql'],
        help='Choose algorithm (default: iql)'
    )
    parser.add_argument(
        '--environment',
        type=str,
        default='kitchen-mixed-v1',
        choices=['kitchen-mixed-v1', 'kitchen-complete-v1',
                 'hopper-medium-v0', 'hopper-expert-v0'],
        help='Choose environment (default: kitchen-mixed-v1)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the saved model'
    )
    return parser.parse_args()


def train_model(args):
    # data=minari.load_dataset('kitchen-complete-v1')
    # dataset, env=get_d4rl('hopper-medium-v0')
    env_name = args.environment
    dataset, env = get_minari(env_name)

    # define evaluators
    td_error_evaluator = metrics.TDErrorEvaluator(
        episodes=dataset.episodes)
    env_evaluator = metrics.EnvironmentEvaluator(env, n_trials=10)
    value_evaluator = metrics.AverageValueEstimationEvaluator(
        episodes=dataset.episodes)

    # define logger
    logger_adapter = logging.CombineAdapterFactory([logging.FileAdapterFactory(root_dir='d3rlpy_test/d3rlpy_logs'),
                                                    logging.TensorboardAdapterFactory(root_dir='d3rlpy_test/tensorboard_logs')])

    # define default encoder
    encoder = VectorEncoderFactory([512, 512, 512])

    # define algorithms
    if args.choose_algorithm == 'cql':
        cql = algos.CQLConfig(actor_learning_rate=5e-5,
                              critic_learning_rate=5e-5,
                              temp_learning_rate=3e-5,
                              actor_encoder_factory=encoder,
                              critic_encoder_factory=encoder,
                              n_action_samples=20,
                              alpha_learning_rate=0,
                              alpha_threshold=5.0,
                              max_q_backup=True,
                              conservative_weight=10.0
                              ).create(device="cuda:0")
        cql.fit(dataset, n_steps=int(5e5),
                n_steps_per_epoch=1000,
                save_interval=50,
                evaluators={'td_error': td_error_evaluator,
                            'environment': env_evaluator, 'value_estimation': value_evaluator},
                logger_adapter=logger_adapter
                )
        cql.save('d3rlpy_test/cql_kitchen_mixed_noalpha.d3')
    elif args.choose_algorithm == 'bc':
        bc = algos.BCConfig(learning_rate=1e-4,
                            encoder_factory=encoder,
                            batch_size=256).create(device="cuda:0")
        bc.fit(dataset, n_steps=int(5e5),
               n_steps_per_epoch=1000,
               save_interval=50,
               evaluators={'environment': env_evaluator},
               logger_adapter=logger_adapter
               )
        bc.save('d3rlpy_test/bc_kitchen_complete.d3')
    elif args.choose_algorithm == 'iql':
        iql_encoder = VectorEncoderFactory([256, 256], dropout_rate=0.1)
        iql = algos.IQLConfig(batch_size=256,
                              weight_temp=0.5,
                              actor_encoder_factory=iql_encoder,
                              critic_encoder_factory=iql_encoder,
                              value_encoder_factory=iql_encoder).create(device="cuda:0")
        iql.fit(dataset, n_steps=int(5e6),
                n_steps_per_epoch=2000,
                save_interval=500,
                evaluators={'td_error': td_error_evaluator,
                            'environment': env_evaluator, 'value_estimation': value_evaluator},
                logger_adapter=logger_adapter
                )
        iql.save(
            'd3rlpy_test/model/iql_kitchen_complete_2ksteps_256batch_256encoder_0.1dropuout_0.5temp.d3')


def test_gymnasiusm_robot():
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make(
        'FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode='human')
    obs, _ = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
    env.close()


def evaluate_model(args, model):
    if args.environment == 'kitchen-mixed-v1':
        gymnasium.register_envs(gymnasium_robotics)
        env = gymnasium.make('FrankaKitchen-v1', render_mode='human',
                             tasks_to_complete=['microwave', 'kettle', 'bottom burner', 'light switch'])
        n_trials = 1
        for _ in range(n_trials):
            obs, _ = env.reset()

            obs_basic = obs['observation']
            obs_microwave = obs['achieved_goal']['microwave']
            obs_kettle = obs['achieved_goal']['kettle']
            obs_bottomburner = obs['achieved_goal']['bottom burner']
            obs_lightswitch = obs['achieved_goal']['light switch']
            # obs_slidecabinet=obs['achieved_goal']['slide cabinet']
            obs = np.concatenate((obs_basic, obs_bottomburner,
                                  obs_kettle, obs_lightswitch, obs_microwave))
            obs = np.expand_dims(obs, axis=0)
            # print(obs)
            while True:
                action = model.predict(obs)[0]
                # print('act: ',action)
                # assert action.shape == (1,9)
                # action=np.squeeze(action)
                # print('act: ',action)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs_basic = obs['observation']
                obs_microwave = obs['achieved_goal']['microwave']
                obs_kettle = obs['achieved_goal']['kettle']
                obs_bottomburner = obs['achieved_goal']['bottom burner']
                obs_lightswitch = obs['achieved_goal']['light switch']
                # obs_slidecabinet=obs['achieved_goal']['slide cabinet']
                obs = np.concatenate(
                    (obs_basic, obs_bottomburner, obs_kettle, obs_lightswitch, obs_microwave))
                obs = np.expand_dims(obs, axis=0)
                # obs=np.expand_dims(obs,axis=0)
                # print('obs: ',obs)
                env.render()
                if terminated or truncated:
                    break
        env.close()
    elif args.environment == 'hopper-medium-v0' or args.environment == 'hopper-expert-v0':
        env = gym.make('Hopper-v3', render_mode='human')
        obs, _ = env.reset()
        obs = np.expand_dims(obs, axis=0)
        for i in range(1000):
            action = model.predict(obs)
            assert action.shape == (1, 3)
            action = np.squeeze(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.expand_dims(obs, axis=0)
            env.render()
        env.close()


def main():
    args = parse_args()
    print(args)
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test_env':
        test_gymnasiusm_robot()
    elif args.mode == 'evaluate':
        model = load_learnable(args.model_path)
        evaluate_model(args, model)
    # iql=load_learnable('/home/zxr/lxy/d4rl/d3rlpy_test/model/iql_kitchen_mixed.d3')
    # ans=metrics.evaluate_qlearning_with_environment(iql,env)
    # print(ans)
    # test_kitchen_model(iql)


if __name__ == '__main__':
    main()