package util

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type Locust struct {
	locustUrl  string
	client     http.Client
	requestUrl string
}

func NewLocust(locustUrl string, client http.Client, requestUrl string) *Locust {
	return &Locust{locustUrl: locustUrl, client: client, requestUrl: requestUrl}
}
func (l *Locust) Client() http.Client {
	return l.client
}
func (l *Locust) SetClient(client http.Client) {
	l.client = client
}
func (l *Locust) RequestUrl() string {
	return l.requestUrl
}
func (l *Locust) SetRequestUrl(requestUrl string) {
	l.requestUrl = requestUrl
}
func (l *Locust) LocustUrl() string {
	return l.locustUrl
}
func (l *Locust) SetLocustUrl(locustUrl string) {
	l.locustUrl = locustUrl
}
func (l *Locust) StartLocust(spawn_rate int, user_count int) {
	url := l.locustUrl + "/swarm"
	method := "POST"
	payload := strings.NewReader("user_count=" + strconv.Itoa(user_count) + "&spawn_rate=" + strconv.Itoa(spawn_rate) + "&host=" + l.requestUrl)
	req, err := http.NewRequest(method, url, payload)
	if err != nil {
		fmt.Println(err)
		return
	}
	req.Header.Add("User-Agent", "Apifox/1.0.0 (https://apifox.com)")
	req.Header.Add("Accept", "*/*")
	req.Header.Add("Connection", "keep-alive")
	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	res, err := l.client.Do(req)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer res.Body.Close()

	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(body))
	println("启动成功")
}
func (l *Locust) ObtainRPS() string {
	// 判断状态
	resp2, err := l.client.Get(l.locustUrl + "/stats/requests")
	if err != nil {
		return "-1"
	}
	body2, err := ioutil.ReadAll(resp2.Body)
	if err != nil {
		return "-1"
	}
	text := string(body2)
	// 使用字符串分割获取字段值
	fields := strings.Split(text, ",")
	for _, field := range fields {
		if strings.Contains(field, "total_rps") {
			parts := strings.Split(field, ":")
			if len(parts) == 2 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "-1"
}
func (l *Locust) StopLocust() {

	count := 0
	for {
		if count > 10 {
			println("停止失败")
			break
		}
		count += 1
		println("停止中")
		resp, err := l.client.Get(l.locustUrl + "/stop")
		if err != nil {
			fmt.Println("Error:", err)
			return
		}
		_, err = ioutil.ReadAll(resp.Body)
		if err != nil {
			continue
		}
		resp.Body.Close()
		if resp.StatusCode == 200 {
			rps := l.ObtainRPS()
			println(rps)
			if rps == "0" || rps == "0.0" {
				break

			} else {
				l.StartLocust(1, 1)

				l.client.Get(l.locustUrl + "/stop")
			}
		}
		time.Sleep(1 * time.Second)
	}
	println("停止成功")

}
func (l *Locust) ObtainWorker() {

}
