### yaml重新排版
* 使用yamllint檢查YAML格式
yamllint 是專門用於檢查 YAML 文件結構和縮排的工具
1. 安裝yamllint
```
$ sudo apt update
$ sudo apt install yamllint
```

2. 檢查 YAML 文件格式
使用以下命令檢查你的 cam_to_cam.yaml 文件是否有縮排或語法錯誤:
```
$ yamllint cam_to_cam.yaml
```

3. 安裝 Node.js LTS (最新穩定版)
```
#添加 Node.js 官方倉庫:
$ curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -

#安裝 Node.js:
$ sudo apt-get install -y nodejs

#驗證 Node.js 版本 (應該是 18.x 以上):
$ node -v
```
4. 重新安裝 prettier
確保已安裝 npm (Node.js 包管理工具):
```
$ sudo apt-get install npm
```

使用 npm 安裝或更新 prettier:
```
$ sudo npm install -g prettier
```

確認 prettier 安裝成功:
```
$ prettier --version
```

5. 格式化 YAML 文件
執行以下命令來自動修復縮排和語法錯誤:
```
$ prettier --write cam_to_cam.yaml
```
6. 最後於yaml檔開頭補上yaml格式宣告
```
%YAML:1.0
---
```
**這將會覆蓋原有文件並修復所有的縮排問題**
