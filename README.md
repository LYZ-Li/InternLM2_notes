![](https://private-user-images.githubusercontent.com/25839884/348517003-93ff2412-777c-4619-812b-0134eb327cf3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjA5NDM5NDksIm5iZiI6MTcyMDk0MzY0OSwicGF0aCI6Ii8yNTgzOTg4NC8zNDg1MTcwMDMtOTNmZjI0MTItNzc3Yy00NjE5LTgxMmItMDEzNGViMzI3Y2YzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA3MTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNzE0VDA3NTQwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA5OTY3ZWNlYTAxMGQzNjEwODUyZGFkNzI4NmE0OTc3MWU0OGM5Yzk5ZjA1ZWIzZjI4NWVhOGI2ZGU4NzQ0YTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Lewhz8MeZ4TCiV6nYzgLTs8zLO6STsFn80O2wqlsCf4)
教程链接  
[GitHub 仓库](https://github.com/InternLM/Tutorial) https://github.com/InternLM/Tutorial


你可以使用 Git 在不下载已有分支文件的情况下，将新文件夹的内容推送到一个已存在仓库的新的分支中。以下是具体步骤：

步骤一：初始化本地仓库
在新文件夹中初始化 Git 仓库：

```sh
cd path/to/your/new/folder
git init
```
添加远程仓库：

```sh
git remote add origin https://github.com/yourusername/yourrepository.git
```
步骤二：创建并切换到新分支
创建并切换到新分支：
```sh
git checkout -b new-branch-name
```
步骤三：添加文件并提交
添加新文件夹中的所有内容：

```sh
git add .
```
提交更改：

```sh
git commit -m "Initial commit for new branch"
```
步骤四：推送新分支到远程仓库
推送新分支到远程仓库：
```sh
git push origin new-branch-name
```