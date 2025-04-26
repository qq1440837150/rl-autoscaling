import logging
import argparse
import threading
import time

import pandas as pd
import schedule
from setuptools.command.alias import alias
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO

from mywork_vpa2.dqn_gym.envs import MyEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.locustOperate import start_locust
from mywork_vpa2.policy.ssh.timeupdate import execute_sudo_command
# Logging
from envs.util import test_model

logging.basicConfig(filename='run.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Run ILP!')
parser.add_argument('--alg', default='a2c', help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')
parser.add_argument('--k8s', default=True, action="store_true", help='K8s mode')
parser.add_argument('--use_case', default='myTest', help='Apps: ["myTest"]')
parser.add_argument('--goal', default='cost', help='Reward Goal: ["cost", "latency"]')

parser.add_argument('--training', default=True, action="store_true", help='Training mode')
parser.add_argument('--testing', default=False, action="store_true", help='Testing mode')
parser.add_argument('--loading', default=True, action="store_true", help='Loading mode')
parser.add_argument('--load_path',
                    default='D:\\python\\gym\\gym-hpa\\mywork_vpa2\\policy\\logs\\a2c_env_test_one_app_vpa_gym2_goal_cost_k8s_True_totalSteps_10000\\a2c_env_test_one_app_vpa_gym2_goal_cost_k8s_True_totalSteps_10000_10000_steps.zip',
                    help='Loading path, ex: logs/model/test.zip')
parser.add_argument('--test_path', default='logs/model/test.zip', help='Testing path, ex: logs/model/test.zip')

parser.add_argument('--steps', default=100, help='The steps for saving.')
parser.add_argument('--total_steps', default=10000, help='The total number of steps.')

args = parser.parse_args()


def get_model(alg, env, tensorboard_log):
    model = 0
    if alg == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    elif alg == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)  # , n_steps=steps
    else:
        logging.info('Invalid algorithm!')

    return model


def get_load_model(alg, tensorboard_log, load_path):
    if alg == 'ppo':
        return PPO.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log, n_steps=500)
    elif alg == 'recurrent_ppo':
        return RecurrentPPO.load(load_path, reset_num_timesteps=False, verbose=1,
                                 tensorboard_log=tensorboard_log)  # n_steps=steps
    elif alg == 'a2c':
        return A2C.load(load_path, reset_num_timesteps=False, verbose=1, tensorboard_log=tensorboard_log)
    else:
        logging.info('Invalid algorithm!')


def get_env(use_case, k8s, goal):
    env = 0
    if use_case == 'myTest':
        env = MyEnv.TestOneApp(k8s=k8s, goal_reward=goal)
    else:
        logging.error('Invalid use_case!')
        raise ValueError('Invalid use_case!')

    return env


def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    k8s = args.k8s
    use_case = args.use_case
    goal = args.goal

    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal)

    scenario = ''
    if k8s:
        scenario = 'real'
    else:
        scenario = 'simulated'

    tensorboard_log = "../../results/" + use_case + "/" + scenario + "/" + goal + "/"

    name = alg + "_env_" + env.name + "_goal_" + goal + "_k8s_" + str(k8s) + "_totalSteps_" + str(total_steps)

    # callback
    checkpoint_callback = CheckpointCallback(save_freq=steps, save_path="logs/" + name, name_prefix=name)

    if training:
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)
        else:
            model = get_model(alg, env, tensorboard_log)
            model.learn(total_timesteps=total_steps, tb_log_name=name + "_run", callback=checkpoint_callback)

        model.save(name)

    if testing:
        model = get_load_model(alg, tensorboard_log, test_path)
        test_model(model, env, n_episodes=100, n_steps=110, smoothing_window=5, fig_name=name + "_test_reward.png")


def myjob():
    hostname = '172.31.234.114'  # 远程服务器的主机名或 IP 地址
    port = 22  # SSH 端口，通常为 22
    username = 'k8s'  # 用户名
    password = 'Admin@9000'  # 密码
    command = 'ntpdate 172.31.73.167'  # 要在远程服务器上执行的命令
    sudo_password = 'Admin@9000'  # sudo 密码
    execute_sudo_command(hostname, port, username, password, sudo_password, command)


myjob()
schedule.every(30).minutes.do(myjob)
curUser = 1
negFlag = 1


def updateUser():
    global curUser
    global negFlag
    print(f"user:{curUser}")
    curUser += 200 * negFlag
    start_locust(curUser, curUser)
    if curUser >= 2000:
        negFlag = -1
    if curUser <= 1:
        negFlag = 1


# def update_dataset():
#     df = pd.read_csv('generateusers3.csv')
#     while True:
#         for u in df["users"]:
#             u = int(u)
#             print(f"user:{u}")
#             start_locust(u, u)
#             time.sleep(60)
#
#
# threading.Thread(target=update_dataset).start()
updateUser()
schedule.every(5).minutes.do(updateUser)


def run_scheduler():
    """函数用于运行调度器，在一个独立的线程中运行"""
    while True:
        schedule.run_pending()
        time.sleep(1)


# 启动调度线程
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

if __name__ == "__main__":
    main()
