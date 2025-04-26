import http.client
import json
import logging
import math
import random
import time

import numpy as np
import pandas as pd
import requests
from kubernetes import client, config

import soft_model.usexgb
from mywork_vpa2.dqn_gym.editcontent import updateOne
from mywork_vpa2.dqn_gym.envs.computeMaxThreads import printMaxThreads
from mywork_vpa2.policy.updateThread import update_thread

# Constants
MAX_CPU = 10000  # cpu in m
MAX_TRAFFIC = 20000
MAX_RESPONSE_TIME = 10000

# port-forward in k8s cluster
# prometheus 地址
PROMETHEUS_URL = 'http://10.111.168.169:9090/'

# Endpoint of your Kube cluster: kube proxy enabled
# Host代理地址
HOST = "http://localhost:8080"

# TODO: Add the TOKEN from your cluster!
TOKEN = ""


def my_deployment_list(k8s, min, max):
    deployment_list = [
        DeploymentStatus(k8s, "testone", "testthread", "testone", "172.31.234.111:5000/testone",
                         max, min, 0.75),
    ]
    return deployment_list


class DeploymentStatus:  # Deployment Status (Workload)
    ## 获取对应的pod和对应的pod uuid
    def __init__(self, k8s, name, namespace, container_name, container_image, max_cpu, min_cpu, threshold=0.75):
        self.throughput = 0
        self.goodput = 0
        self.request_cpu = min_cpu

        self.cpu_utilization = 0
        self.name = name
        # namespace
        self.namespace = namespace
        # container_name
        self.container_name = container_name
        # container image
        self.container_image = container_image

        # CPU & MEM threshold
        self.threshold = threshold

        # Pod Names
        self.pod_names = ["pod-1"]
        # MAX Number of Pods
        self.max_cpu = max_cpu
        # MIN Number of Pods
        self.min_cpu = min_cpu
        # Number of Pods
        self.cur_cpu = min_cpu  # Initialize as 1
        # Number of Pods in previous step
        self.previous_cpu = min_cpu  # Initialize as 1
        ## 垂直伸缩，副本数量固定为2
        self.num_pods = 2

        # CPU Target (in m)
        self.cpu_target = int(self.threshold * self.cur_cpu)

        self.MAX_CPU = MAX_CPU  # cpu in m

        # CPU Usage Aggregated (in m)
        self.cur_cpu = random.randint(1, 5000)

        self.cpu_usage = random.randint(1, 5000)  # sample['cpu'].values[0]

        # Current Requests
        self.received_traffic = random.randint(1, get_max_traffic())  # sample['traffic_in'].values[0]
        self.transmit_traffic = random.randint(1, get_max_traffic())  # sample['traffic_out'].values[0]

        # Throughput PING INLINE
        # self.ping = 0

        # K8s enabled?
        self.k8s = k8s

        # csv file
        self.csv = self.namespace + '_' + self.name + '.csv'

        # time between API calls if failure happens
        self.sleep = 0.2

        # App. Latency
        self.latency = 0

        if self.k8s:  # Real env: consider a k8s cluster
            logging.info("[Deployment] Consider a real k8s cluster ... ")
            self.token = TOKEN

            # Create a configuration object
            self.config = client.Configuration()
            self.config.verify_ssl = False
            self.config.api_key = {"authorization": "Bearer " + self.token}

            # Specify the endpoint of your Kube cluster: kube proxy enabled
            config.load_kube_config()

            # Create a ApiClient with our config
            self.client = client.ApiClient()

            # v1 api
            self.v1 = client.CoreV1Api()
            # apps v1 api
            self.apps_v1 = client.AppsV1Api()
            self.obtainPodAndUUID()

            self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
            self.update_pods(self.cur_cpu)
            # Update the limits of Pods
            self.cur_cpu = self.obtainPodRequest()
            self.previous_cpu = self.cur_cpu
            # update obs
            self.update_obs_k8s()

    def update_obs_k8s(self):
        self.cpu_usage = 0
        self.mem_usage = 0
        self.received_traffic = 0
        self.transmit_traffic = 0

        # Previous number of Pods
        self.previous_cpu = self.obtainPodRequest()

        # Get deployment object
        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
        # Update number of Pods
        self.cur_cpu = self.obtainPodRequest()
        curtime = int(time.time())
        time.sleep(1)
        gp, all, self.meanrespnsetime, throughputRatio = self.obtainMetric2(curtime - 2, curtime, 100)
        self.throughput = all
        self.goodput = gp
        print(f"gp:{gp},all:{all},mean:{self.meanrespnsetime}")

        # logging.info("[Update obs] Current Pods: " + str(self.num_pods))

        # Get received / transmit traffic
        for p in self.pod_names:
            query_cpu = 'sum(rate(container_cpu_usage_seconds_total{namespace=' \
                        '"' + self.namespace + '", pod="' + p + '",container="testone"}[10s])) by (pod)'

            query_mem = 'sum(irate(container_memory_working_set_bytes{namespace=' \
                        '"' + self.namespace + '", pod="' + p + '"}[5m])) by (pod)'

            query_received = 'sum(rate(container_network_receive_bytes_total{namespace=' \
                             '"' + self.namespace + '", pod="' + p + '"}[30s])) by (pod)'
            query_transmit = 'sum(rate(container_network_transmit_bytes_total{namespace="' \
                             + self.namespace + '", pod="' + p + '"}[30s])) by (pod)'

            # -------------- CPU ----------------
            results_cpu = self.fetch_prom(query_cpu)
            repeat_count = 0
            if results_cpu:
                cpu = int(float(results_cpu[0]['value'][1]) * 1000)  # saved as m
                while cpu == 0 and repeat_count < 20:
                    repeat_count += 1
                    try:
                        results_cpu = self.fetch_prom(query_cpu)
                        cpu = int(float(results_cpu[0]['value'][1]) * 1000)  # saved as m
                    except Exception:
                        cpu = 0
                    time.sleep(1)

                self.cpu_usage += cpu

            # -------------- MEM ----------------
            results_mem = self.fetch_prom(query_mem)
            if results_mem:
                mem = int(float(results_mem[0]['value'][1]) / 1000000)  # saved as Mi
                self.mem_usage += mem

            # -------------- Received Traffic  ----------------
            results_received = self.fetch_prom(query_received)
            if results_received:
                rec = int(float(results_received[0]['value'][1]))
                rec = int(rec / 1000)  # saved as KBit/s
                self.received_traffic += rec

            # -------------- Transmit Traffic  ----------------
            results_transmit = self.fetch_prom(query_transmit)
            if results_transmit:
                trans = int(float(results_transmit[0]['value'][1]))
                trans = int(trans / 1000)  # saved as KBit/s
                self.transmit_traffic += trans

        # Update Desired replicas
        self.update_limits()

        return

    def updateThreadOnline(self, min, max, step):
        df = pd.DataFrame(columns=['users', 'threads', 'gp', 'all', 'qos'])
        for th in range(min, max + 1, step):
            for count in range(10):
                for ip in self.pod_ip:
                    update_thread(ip, "minSpareThreads", 1)
                    update_thread(ip, "maxThreads", th)
                time.sleep(0.1)
                ## 这里就可以获取gp和all了
                curTime = int(time.time())
                gp, all, self.meanrespnsetime, throughputRatio = self.obtainMetric2(curTime, curTime, 200)
                row = {
                    "users": "1000",
                    "threads": th,
                    "gp": gp,
                    "all": all,
                    "qos": 200
                }
                df = df._append(row, ignore_index=True)
        computThreads = printMaxThreads(df, min, max, step)
        print("在线搜索的结果：" + str(computThreads))
        return computThreads

    def update_limits(self):
        self.cpu_utilization = self.cpu_usage / (self.cur_cpu * self.num_pods)
        if self.cpu_utilization > 1:
            self.cpu_utilization = 1
        return

    def update_soft(self):
        max_thread = soft_model.usexgb.predictor.predict([self.cur_cpu])[0]
        print("max_thread:" + str(max_thread))
        max_thread = int(max_thread)
        for ip in self.pod_ip:
            update_thread(ip, "minSpareThreads", 1)
            update_thread(ip, "maxThreads", max_thread)
        return max_thread

    def localhostSearch(self, predict):
        # 这种情况下没有任何的请i去，不需要任何处理
        if self.throughput == 0:
            return predict
        is_previous = True
        ## 先获取当前的gp/all
        gpt = self.goodput / self.throughput
        if gpt < 0.99:
            # 往前搜索
            min_value = max(1, predict - 3)
            max_value = max(1, predict - 1)
            is_previous = True
        else:
            # 往后面搜索
            min_value = min(predict + 1, 200)
            max_value = min(predict + 3, 200)
            is_previous = False

        ## todo； 如果搜索到最后一个，都还是满足，
        ## todo:    那么考虑当前模型的线程数量是不是太低了，那么可以获取cpu使用率，如果cpu使用率低于80%，那么就是太低了，更新为最后一个线程
        ## todo:    那么也有科恩那个是当前的资源请求压力不大，那么可以使用调整前后的吞吐量是否又明显提升，例如120%那么就说明确实太低了，更新为最后一个
        ## todo： 如果搜索到第一个，就说明当前线程太高了，直接更新
        update_value = 0
        for th in range(min_value, max_value + 1):
            gplist = []
            alllist = []
            gpdieall = []
            update_value = th
            for count in range(5):
                for ip in self.pod_ip:
                    update_thread(ip, "minSpareThreads", 1)
                    update_thread(ip, "maxThreads", th)
                time.sleep(0.1)
                ## 这里就可以获取gp和all了
                curTime = int(time.time())
                gp, all, self.meanrespnsetime, throughputRatio = self.obtainMetric2(curTime, curTime, 100)
                gplist.append(gp)
                alllist.append(all)
                if all != 0:
                    gpdieall.append(gp / all)
            mean_sat = np.mean(gpdieall)
            std_dev = np.std(gpdieall, ddof=1)
            n = len(gpdieall)
            if n == 0:
                continue
            confidence_interval = (mean_sat - 1.96 * (std_dev / np.sqrt(n)), mean_sat + 1.96 * (std_dev / np.sqrt(n)))
            if confidence_interval[0] < 0.96:
                ## 更新模型
                soft_model.usexgb.updateModel(self.cur_cpu, th)
                break
            if is_previous == False and th == max_value:
                ## 再获取一次利用率
                curTime = int(time.time())
                gp, all, self.meanrespnsetime, throughputRatio = self.obtainMetric2(curTime - 2, curTime, 100)
                if all > self.throughput * 1.1 and self.cpu_usage < 0.8:
                    ## 更新模型
                    soft_model.usexgb.updateModel(self.cur_cpu, th)
                    pass
                else:
                    update_value = predict
                    for ip in self.pod_ip:
                        update_thread(ip, "minSpareThreads", 1)
                        update_thread(ip, "maxThreads", predict)
                    pass
        return update_value

    def update_soft_certain(self, threads_num):
        for ip in self.pod_ip:
            update_thread(ip, "minSpareThreads", 1)
            update_thread(ip, "maxThreads", threads_num)

    def fetch_prom(self, query):
        try:
            response = requests.get(PROMETHEUS_URL + '/api/v1/query',
                                    params={'query': query})

        except requests.exceptions.RequestException as e:
            print(e)
            print("Retrying in {}...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        if response.json()['status'] != "success":
            print("Error processing the request: " + response.json()['status'])
            print("The Error is: " + response.json()['error'])
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        result = response.json()['data']['result']
        return result

    def print_deployment(self):
        logging.info("[Deployment] Name: " + str(self.name))
        logging.info("[Deployment] Namespace: " + str(self.namespace))
        logging.info("[Deployment] Number of pods: " + str(self.num_pods))
        logging.info("[Deployment] Pod Names: " + str(self.pod_names))
        logging.info("[Deployment] MAX Cpu: " + str(self.max_cpu))
        logging.info("[Deployment] MIN Cpu: " + str(self.min_cpu))
        logging.info("[Deployment] CPU Usage (in m): " + str(self.cpu_usage))
        logging.info("[Deployment] MEM Usage (in Mi): " + str(self.mem_usage))
        logging.info("[Deployment] Received traffic (in Kbit/s): " + str(self.received_traffic))
        logging.info("[Deployment] Transmit traffic (in Kbit/s): " + str(self.transmit_traffic))
        logging.info("[Deployment] latency (in ms): " + str(self.latency))

    def update_deployment(self, new_cpu):
        # Get deployment object
        self.deployment_object = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
        # logging.info(self.deployment_object)

        # Update previous number of pods
        self.previous_cpu = obtain_deploy_request_cpu(self.deployment_object)

        # Update replicas
        self.deployment_object.spec.template.spec.containers[0].resources.requests["cpu"] = str(new_cpu) + "m"
        self.deployment_object.spec.template.spec.containers[0].resources.limits["cpu"] = str(new_cpu * 1.2) + "m"

        # try to patch the deployment
        self.patch_deployment(new_cpu)

    def update_pods(self, new_cpu):
        self.updateAnnotation(new_cpu)
        for pod_uid in self.uuids:
            pod_uid = pod_uid.replace("-", "_")
            updateOne(pod_uid, int(new_cpu) * 100)

    def patch_deployment(self, new_cpu):
        try:
            self.apps_v1.patch_namespaced_deployment(
                name=self.name, namespace=self.namespace, body=self.deployment_object
            )
        except Exception as e:
            print(e)
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.update_deployment(new_cpu)

    def updateAnnotation(self, cpu):
        # 合并新的 annotations
        self.pods[0].metadata.annotations.update({
            "request_cpu": str(cpu)
        })

        # 创建一个 Patch 字典来更新 Pod
        body = {"metadata": {"annotations": self.pods[0].metadata.annotations}}
        self.v1.patch_namespaced_pod(name=self.pods[0].metadata.name, namespace=self.namespace,
                                     body=body)

    def deploy_pod_replicas_cpu(self, n, env):
        # Deploy pods if possible
        cpu = self.cur_cpu + n

        # logging.info("Deployment name: " + str(self.name))
        # logging.info("Current replicas: " + str(self.num_pods))
        # logging.info("New replicas: " + str(replicas))

        if cpu <= self.max_cpu:
            # logging.info("[Take Action] Add {} Replicas".format(str(n)))
            if self.k8s:  # patch deployment on k8s cluster
                self.update_pods(cpu)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = cpu
            return
        else:
            # logging.info("Constraint: MAX Pod Replicas! Desired replicas: " + str(replicas))
            env.constraint_max_pod_cpu = True

    def terminate_pod_replicas_cpu(self, n, env):
        # Terminate pods if possible
        cpu = self.cur_cpu - n

        # logging.info("Deployment name: " + str(self.name))
        # logging.info("Current replicas: " + str(self.num_pods))
        # logging.info("New replicas: " + str(replicas))

        if cpu >= self.min_cpu:
            # logging.info("[Take Action] Terminate {} Replicas".format(str(n)))
            if self.k8s:  # patch deployment on k8s cluster
                self.update_pods(cpu)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = cpu
            return
        else:
            # logging.info("Constraint: MIN Pod Replicas! Desired replicas: " + str(replicas))
            env.constraint_min_pod_cpu = True

    def obtainMetric2(self, start, end, qos):
        err = 0
        while err < 10:
            try:
                conn = http.client.HTTPConnection("10.108.212.226", 8080, timeout=1)
                conn.request("GET", f'/metric/obtainMetric?start={start}&end={end}&qos={qos}')
                res = conn.getresponse()
                data = res.read()
                jsonobj = json.loads(data.decode("utf-8"))
                return jsonobj['gp'], jsonobj['all'], jsonobj['meanReponseTime'], jsonobj['throughputRatio']
            except Exception as e:
                err += 1
                time.sleep(1)
        return 0, 0, 0, 0

    def obtainPodAndUUID(self):
        self.pods = []
        self.uuids = []
        self.pod_names = []
        self.pod_ip = []
        # 获取 Deployment 的标签选择器
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
            label_selector = deployment.spec.selector.match_labels
            selector_string = ",".join([f"{key}={value}" for key, value in label_selector.items()])

            # 根据标签选择器获取匹配的 Pod
            pods = self.v1.list_namespaced_pod(namespace=self.namespace, label_selector=selector_string)
            for p in pods.items:
                if p.metadata.labels['app'] == self.name:
                    self.pod_names.append(p.metadata.name)
            # 输出所有 Pod 的名称
            self.pods = pods.items
            self.uuids = [pod.metadata.uid for pod in pods.items]
            self.pod_ip = [pod.status.pod_ip for pod in pods.items]
        except client.exceptions.ApiException as e:
            print(f"Error retrieving deployment or pods: {e}")

    def obtainPodRequest(self):
        annotations = self.pods[0].metadata.annotations
        if "request_cpu" in annotations:
            request_cpu = self.pods[0].metadata.annotations["request_cpu"]
            self.request_cpu = int(request_cpu)
        else:
            self.request_cpu = self.min_cpu

        if self.request_cpu < self.min_cpu:
            self.request_cpu = self.min_cpu
        elif self.request_cpu > self.max_cpu:
            self.request_cpu = self.max_cpu
        return self.request_cpu


def get_max_cpu():
    return MAX_CPU


def get_max_traffic():
    return MAX_TRAFFIC


def get_max_response_time():
    return MAX_RESPONSE_TIME


def convert_to_milli_cpu(value):
    new_value = int(value[:-1])
    if value[-1] == "n":
        new_value = int(value[:-1])
        new_value = int(new_value / 1000000)

    return new_value


def change_usage(min, max, max_threshold):
    if max > max_threshold:
        max = max_threshold

    if min < 0:
        min = 0

    return random.randint(min, max)


def convert_to_mega_memory(value):
    last_two = value[-2:]
    new_value = 0

    if last_two == "Ki":
        size = len(value)
        # Slice string to remove last 2 characters
        new_value = int(value[:size - 2])
        new_value = int(new_value / 1000)

    return new_value


def obtain_deploy_limit_cpu(deploy):
    container = deploy.spec.template.spec.containers[0]
    resource_limits = container.resources.limits if container.resources and container.resources.limits else {}
    return parse_cpu_limit_to_milli(resource_limits.get("cpu"))


def obtain_deploy_request_cpu(deploy):
    container = deploy.spec.template.spec.containers[0]
    resource_requests = container.resources.requests if container.resources and container.resources.requests else {}
    return parse_cpu_limit_to_milli(resource_requests.get("cpu"))


def parse_cpu_limit_to_milli(cpu_limit):
    if cpu_limit is None:
        return None
    elif cpu_limit.endswith("m"):
        # 如果是毫核，直接去掉 'm' 转换为整数
        return int(float(cpu_limit[:-1]))
    else:
        # 如果是核单位，乘以 1000 转换为毫核
        return int(float(cpu_limit) * 1000)
