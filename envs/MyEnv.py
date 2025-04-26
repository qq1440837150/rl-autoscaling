import csv
import datetime
import math
from datetime import datetime
import logging
import time
from statistics import mean

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

# Number of Requests - Discrete Event
from mywork_vpa2.dqn_gym.envs.deployment import get_max_cpu, get_max_traffic, my_deployment_list, get_max_response_time
from mywork_vpa2.dqn_gym.envs.util import get_pods_cpu, get_cost_reward, save_to_csv, get_cost_reward2

MAX_STEPS = 25  # MAX Number of steps per episode

ACTION_DO_NOTHING = 0
ACTION_ADD_100_REPLICA = 1
ACTION_ADD_200_REPLICA = 2
ACTION_ADD_300_REPLICA = 3
ACTION_ADD_400_REPLICA = 4
ACTION_ADD_500_REPLICA = 5
ACTION_TERMINATE_100_REPLICA = 6
ACTION_TERMINATE_200_REPLICA = 7
ACTION_TERMINATE_300_REPLICA = 8
ACTION_TERMINATE_400_REPLICA = 9
ACTION_TERMINATE_500_REPLICA = 10

DEPLOYMENTS = ["testone"]
# Action Moves
MOVES = ["None", "Add-100", "Add-200", "Add-300", "Add-400", "Add-500", "Add-600", "Add-700", "Add-800", "Add-900",
         "Add-1000",
         "Stop-100", "Stop-200", "Stop-300", "Stop-400", "Stop-500", "Stop-600", "Stop-700", "Stop-800", "Stop-900",
         "Stop-1000"]
# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_testone = 0

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class TestOneApp(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward="cost", waiting_period=1):
        super(TestOneApp, self).__init__()
        self.k8s = k8s
        self.name = "test_one_app_vpa_gym2"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.waiting_period = waiting_period  # seconds to wait after action
        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))
        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 11
        # Multi-Discrete
        # Deployment: Discrete 11
        # Action: Discrete 9 - None[0], Add-1[1], Add-2[2], Add-3[3], Add-4[4],
        #                      Stop-1[5], Stop-2[6], Stop-3[7], Stop-4[8]

        self.action_space = spaces.MultiDiscrete([1, self.num_actions])
        # Observations: 22 Metrics! -> 2 * 11 = 22
        # "number_pods"                     -> Number of deployed Pods
        # "cpu_usage_aggregated"            -> via metrics-server
        # "mem_usage_aggregated"            -> via metrics-server
        # "cpu_requests"                    -> via metrics-server/pod
        # "mem_requests"                    -> via metrics-server/pod
        # "cpu_limits"                      -> via metrics-server
        # "mem_limits"                      -> via metrics-server
        # "lstm_cpu_prediction_1_step"      -> via pod annotation
        # "lstm_cpu_prediction_5_step"      -> via pod annotation
        # "average_number of requests"      -> Prometheus metric: sum(rate(http_server_requests_seconds_count[5m]))

        self.min_cpu = 100
        self.max_cpu = 10000
        self.deploymentList = my_deployment_list(self.k8s, self.min_cpu, self.max_cpu)
        # Logging Deployment
        for d in self.deploymentList:
            d.print_deployment()

        self.observation_space = self.get_observation_space()
        # Info
        self.total_reward = None
        self.avg_cpus = []
        self.avg_latency = []
        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_pod_cpu = False
        self.constraint_min_pod_cpu = False
        self.cost_weight = 0  # add here a value to consider cost in the reward function

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"
        # self.df = pd.read_csv("../../datasets/real/" + self.deploymentList[0].namespace + "/v1/"
        #                       + self.name + '_' + 'observation.csv')
        self.testingMode = True

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_cpus = []
        self.avg_latency = []
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        # Deployment Data
        self.deploymentList = my_deployment_list(self.k8s, self.min_cpu, self.max_cpu)
        print("episode reset")

        return np.array(self.get_state())

    def reset2(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_cpus = []
        self.avg_latency = []
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        return np.array(self.get_state())

    def updateThread(self):
        for deployment in self.deploymentList:
            deployment.update_soft_certain(200)

    def startOnlineThread(self):
        for deployment in self.deploymentList:
            result = deployment.updateThreadOnline(1, 200, 10)
            if result == None:
                result = 200
            if result > 0 and result <= 200:
                deployment.update_soft_certain(result)

    def updateMaxThread(self):
        if self.testingMode:
            for deployment in self.deploymentList:
                # 软件伸缩
                predict = deployment.update_soft()
                update_predict = deployment.localhostSearch(predict)
                print("更新前："+str(predict))
                print("更新后："+str(update_predict))

    def step(self, action):
        print("step start")
        self.step_time = time.time()
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()
        # Get first action: deployment
        if action[ID_DEPLOYMENTS] == 0:  # recommendation
            n = ID_testone
        else:  # ==10 email
            n = ID_testone
        # Execute one time step within the environment
        self.take_action(action[ID_MOVES], n)
        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_cpu is False \
                    and self.constraint_max_pod_cpu is False:
                # logging.info('[Step {}] | Waiting {} seconds for enabling action ...'
                # .format(self.current_step, self.waiting_period))
                time.sleep(self.waiting_period)  # Wait a few seconds...

        # Update observation before reward calculation:
        if self.k8s:  # k8s cluster
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:
            self.simulation_update()
        # self.updateMaxThread()
        self.updateThread()

        # Get reward
        reward = self.get_reward
        self.total_reward += reward

        self.avg_cpus.append(get_pods_cpu(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        # Print Step and Total Reward
        # if self.current_step == MAX_STEPS:
        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, DEPLOYMENTS[action[0]], MOVES[action[1]], reward, self.total_reward))
        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[0].latency)

        self.info = dict(
            total_reward=self.total_reward,
        )

        # Update Reward Keywords
        self.constraint_max_pod_cpu = False
        self.constraint_min_pod_cpu = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start

            # logging.info('Avg. latency : {} ', float("{:.3f}".format(mean(self.avg_latency))))
            save_to_csv(self.file_results, self.episode_count, mean(self.avg_cpus), mean(self.avg_latency),
                        self.total_reward, self.execution_time)
        print("step stop")
        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    def get_observation_space(self):
        return spaces.Box(
            low=np.array([
                self.min_cpu,  # Number of Pods  -- 1) testone
                0,  # CPU Usage (in m)
                0,  # Average Number of received traffic
                0,  # Average Number of transmit traffic
                0,
                0,
            ]), high=np.array([
                self.max_cpu,  # Number of Pods -- 1)
                get_max_cpu(),  # CPU Usage (in m)
                get_max_traffic(),  # Average Number of received traffic
                get_max_traffic(),  # Average Number of transmit traffic
                10000,
                10000,
            ]),
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def simulation_update(self):
        pass

    def take_action(self, action, id):
        self.current_step += 1
        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True
        # ACTIONS
        if action == ACTION_DO_NOTHING:
            # logging.info("[Take Action] SELECTED ACTION: DO NOTHING ...")
            pass
        elif action == ACTION_ADD_100_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deploymentList[id].deploy_pod_replicas_cpu(100, self)

        elif action == ACTION_ADD_200_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas_cpu(200, self)

        elif action == ACTION_ADD_300_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas_cpu(300, self)

        elif action == ACTION_ADD_400_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas_cpu(400, self)

        elif action == ACTION_ADD_500_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas_cpu(500, self)
        elif action == ACTION_TERMINATE_100_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deploymentList[id].terminate_pod_replicas_cpu(100, self)

        elif action == ACTION_TERMINATE_200_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas_cpu(200, self)

        elif action == ACTION_TERMINATE_300_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas_cpu(300, self)

        elif action == ACTION_TERMINATE_400_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas_cpu(400, self)

        elif action == ACTION_TERMINATE_500_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas_cpu(500, self)
        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        """ Calculate Rewards """
        # Reward based on Keyword!
        if self.constraint_max_pod_cpu:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -3000  # penalty

        if self.constraint_min_pod_cpu:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -3000  # penalty

        # Reward Calculation
        reward = self.calculate_reward()
        return reward

    def calculate_reward(self):
        # Calculate Number of desired Replicas
        reward = 0
        if self.goal_reward == COST:
            reward = get_cost_reward2(self.deploymentList)
        return reward

    def get_state(self):
        ob = (
            self.deploymentList[ID_testone].cur_cpu,
            self.deploymentList[ID_testone].cpu_usage,
            self.deploymentList[ID_testone].received_traffic,
            self.deploymentList[ID_testone].transmit_traffic,
            self.deploymentList[ID_testone].throughput,
            self.deploymentList[ID_testone].goodput
        )
        return ob

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        with file:
            fields.append('date')
            for d in self.deploymentList:
                fields.append(d.name + '_cur_cpu_limit')
                fields.append(d.name + '_desired_cpu_limit')
                fields.append(d.name + '_cpu_usage')
                fields.append(d.name + '_traffic_in')
                fields.append(d.name + '_traffic_out')
                fields.append(d.name + "_throughput")
            '''
            fields = ['date', 'redis-leader_num_pods', 'redis-leader_desired_replicas', 'redis-leader_cpu_usage', 'redis-leader_mem_usage',
                      'redis-leader_cpu_request', 'redis-leader_mem_request', 'redis-leader_cpu_limit', 'redis-leader_mem_limit',
                      'redis-leader_traffic_in', 'redis-leader_traffic_out',
                      'redis-follower_num_pods', 'redis-follower_desired_replicas', 'redis-follower_cpu_usage',
                      'redis-follower_mem_usage', 'redis-follower_cpu_request', 'redis-follower_mem_request', 'redis-follower_cpu_limit',
                      'redis-follower_mem_limit', 'redis-follower_traffic_in', 'redis-follower_traffic_out']
            '''
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()  # write header

            # TO ALTER!
            # DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice",
            # "adservice", "paymentservice", "shippingservice", "currencyservice",
            # "redis-cart", "checkoutservice", "frontend", "emailservice"]

            writer.writerow(
                {'date': date,
                 'testone_cur_cpu_limit': float("{}".format(obs[0])),
                 'testone_cpu_usage': float("{}".format(obs[1])),
                 'testone_traffic_in': float("{}".format(obs[2])),
                 'testone_traffic_out': float("{}".format(obs[3])),
                 'testone_throughput': float("{:.3f}".format(obs[4])),
                 }
            )
        return


if __name__ == '__main__':
    env = TestOneApp(True, "cost", 1)
    env.reset()
    env.deploymentList[0].cur_cpu = 1400
    env.deploymentList[0].update_pods(1400)
    env.updateMaxThread()
