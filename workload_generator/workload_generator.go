package main

import (
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"scheduling/util"
	"strconv"
	"strings"
	"time"
)

//读取数据集，每5分钟换一次

// 修改locust 和 共享文件

// 收集数据
// 收集这个过程中的 cpu使用率和cpu request和cpu limit
// 收集这个过程中的请求gp数量。
func main() {
	// 打开CSV文件
	file, err := os.Open("D:\\项目\\goland\\scheduling\\test\\testmaxthread\\data\\generateusers11.csv")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	// 创建CSV reader
	reader := csv.NewReader(file)

	// 读取CSV文件中的所有记录
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	client := http.Client{Timeout: 5 * time.Second}
	locust := util.NewLocust("http://10.96.30.118:8089", client, "http://10.105.39.224:8081")
	// 遍历记录并打印每一行
	for _, row := range records {
		//

		users := strings.TrimSpace(row[1])

		userNum, err := strconv.ParseInt(users, 10, 64)
		if err != nil {
			continue
		}
		WriteFile(users)
		// 启动
		locust.StartLocust(int(userNum), int(userNum))
		time.Sleep(1 * time.Minute)
		//locust.StopLocust()
	}
}
func WriteFile(users string) {
	// 写入内容到文件
	data := []byte(users)
	err := ioutil.WriteFile(util.ShareFilePath, data, 0644)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Data written to file successfully.")
}
