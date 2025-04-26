import requests
import time

import deployoperator
import kubernetesOperater

locustUrl = "http://10.96.30.118:8089"
requestUrl = "http://10.105.39.224:8081"

def obtain_rps():
    # 获取 RPS
    try:
        resp = requests.get(locustUrl + "/stats/requests",timeout=1)
        resp.raise_for_status()
        data = resp.json()
        total_rps = data.get("total_rps", "-1")
        return str(total_rps)
    except Exception as e:
        print("Error obtaining RPS:", e)
        return "-1"
def obtain_works():
    # 获取 RPS
    try:
        resp = requests.get(locustUrl + "/stats/requests",timeout=1)
        resp.raise_for_status()
        data = resp.json()
        workers = data.get("workers", [])
        return len(workers)
    except Exception as e:
        print("Error obtaining RPS:", e)
        return -1
def stop_locust():
    count = 0
    while count <= 10:
        count += 1
        print("停止中")
        try:
            resp = requests.get(locustUrl + "/stop")
            resp.raise_for_status()
            reset()
            time.sleep(1)
            rps = obtain_rps()
            if rps == "0" or rps == "0.0":
                print("停止成功")
                break
            else:
                kubernetesOperater.deletePodOfDeployment("locust","testthread")
                time.sleep(10)
                # kubernetesOperater.deletePodOfDeployment("locust-slave","testthread")
                if obtain_works() == 5:
                    print("停止成功")
                    break
        except Exception as e:
            print("Error stopping Locust:", e)
            continue
        time.sleep(1)

def start_locust(spawn_rate, user_count):
    # 启动 Locust
    url = locustUrl + "/swarm"
    payload = {
        "user_count": user_count,
        "spawn_rate": spawn_rate,
        "host": requestUrl
    }
    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    while True:
        try:
            resp = requests.post(url, data=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            print(resp.text)
            print("启动成功")
            break
        except Exception as e:
            print("Error starting Locust:", e)



def reset():
    url = locustUrl + "/stats/reset"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        print(resp.text)
        print("重启成功")
    except Exception as e:
        print("Error starting Locust:", e)
        return e
    return None


# print(obtain_works())
