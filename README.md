# d2l_qlf 
conda activate /home/qlf/anaconda3/envs/d2l

## git
git status
git remote -v 					#查看当前所有远程地址别名
git remote add 别名 远程地址 		# 起别名
git push 别名 远程分支分支 				# 推送本地分支上的内容到远程仓库
git clone 远程地址 				# 将远程仓库的内容克隆到本地
git pull 远程库地址别名 远程分支名 #将远程仓库对于分支最新内容拉下来后与当前本地分支直接合并

vscode git管理push失败，但是终端命令可以成功。

# tmux
https://www.jianshu.com/p/71999b35ead7

//创建session
tmux
//创建并指定session名字
tmux new -s $session_name
//临时退出session
Ctrl+b d
//列出session
tmux ls
//进入已存在的session
tmux a -t $session_name

进入tmux翻屏模式
先按 ctrl ＋ｂ，松开，然后再按 '['

实现上下翻页
进入翻屏模式后，PgUp PgDn 实现上下翻页

退出
q

## clash
cd /home/qlf/clash/clash-for-linux

sudo bash start.sh
source /etc/profile.d/clash.sh
proxy_on
curl www.google.com

sudo bash shutdown.sh
proxy_off

如果需要对Clash配置进行修改，请修改 `conf/config.yaml` 文件。然后运行 `restart.sh` 脚本进行重启。
sudo bash restart.sh
> **注意：**
> 重启脚本 `restart.sh` 不会更新订阅信息。

- 访问 Clash Dashboard
通过浏览器访问 `start.sh` 执行成功后输出的地址，例如：http://localhost:9099/ui
add http://localhost:9099 123
