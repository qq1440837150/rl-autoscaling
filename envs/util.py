import time

import pandas as pd
from matplotlib import pyplot as plt

from soft_model import usexgb


def test_model(model, env, n_episodes, n_steps, smoothing_window, fig_name):
    reward_sum = 0
    obs = env.reset2()
    print("------------Testing -----------------")
    for e in range(n_episodes):
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                print("Episode {} | Total reward: {} |".format(e, str(reward_sum)))
                reward_sum = 0
                print("执行结果："+str(env.current_cpu)+"cpu  "+str(env.getThread())+"thread\n")
                obs = env.reset2()
                break

    env.close()

    # Free memory
    del model, env



def test_model2(model, env, n_steps):
    episode_rewards = []
    reward_sum = 0
    obs = env.reset()

    print("------------Testing -----------------")
    # env.updateThread()
    # env.startOnlineThread()
    while True:
        for _ in range(n_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                episode_rewards.append(reward_sum)
                print("Total reward: {} |".format(str(reward_sum)))
                reward_sum = 0
                obs = env.reset2()
                # env.startOnlineThread()
                usexgb.saveModel()
                break

