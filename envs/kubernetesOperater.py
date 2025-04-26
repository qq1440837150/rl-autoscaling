from kubernetes import client, config
import time

def obtain_pod_ip():
    config.load_kube_config()  # 加载 kubeconfig 文件

    v1 = client.CoreV1Api()

    while True:
        try:
            pod = v1.read_namespaced_pod("testone-pod", "testthread")
            if pod.status.pod_ip:
                print(pod.status.pod_ip)
                return pod.status.pod_ip
        except client.ApiException as e:
            print("Error reading pod:", e)

        time.sleep(1)

    return ""

def deletePod(podname,namespace):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    v1.delete_namespaced_pod(podname,namespace)

def deletePodOfDeployment(deployment,namespace):
    # 加载 kubeconfig 文件
    config.load_kube_config()
    # 创建 Kubernetes API 客户端对象
    api_instance = client.AppsV1Api()
    core_api = client.CoreV1Api()
    try:
        # 读取指定命名空间中的 Deployment
        deployment = api_instance.read_namespaced_deployment(name=deployment, namespace=namespace)

        # 获取 Deployment 的标签选择器
        selector = deployment.spec.selector.match_labels
        label_selector = ','.join([f'{key}={value}' for key, value in selector.items()])

        # 使用标签选择器检索该 Deployment 的所有 Pod
        pod_list = core_api.list_namespaced_pod(namespace, label_selector=label_selector)

        # 删除 Pod
        for pod in pod_list.items:
            core_api.delete_namespaced_pod(pod.metadata.name, namespace)
            print("Deleted Pod:", pod.metadata.name)


    except Exception as e:
        print("Error:", e)


# deletePodOfDeployment("locust-slave","testthread")