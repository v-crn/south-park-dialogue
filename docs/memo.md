# memo

起動

`uvicorn [実行ファイルの親ディレクトリ].[実行ファイル名]:app --reload`

```
uvicorn src.main:app --reload
```

もしくは

```
pipenv run python src/app.py
```

## Docker

### イメージ作成

`docker build -t [イメージ名] [Dockerfileのパス]`

```
docker build -t pipenv_docker .
```

確認

```
docker images
```

### イメージ削除

```
docker rmi [イメージID] --force
```

### コンテナ起動

```
docker run -it pipenv_docker
```

ディレクトリ構成

```py
>>> import os 
>>> os.getcwd()
'/app'
>>> os.listdir("./")
['.venv', 'Pipfile', 'models', 'src', '.DS_Store', 'Pipfile.lock', 'docs', 'Dockerfile', 'data']
>>> os.listdir("../")
['home', 'root', 'lib', 'proc', 'sys', 'boot', 'sbin', 'mnt', 'srv', 'lib64', 'media', 'etc', 'opt', 'bin', 'usr', 'dev', 'var', 'run', 'tmp', '.dockerenv', 'app']
```

コンテナへログインしてbashで操作（起動中のコンテナのみ）

`docker exec -i -t [コンテナID] bash`

```
docker exec -i -t 83b400d0295b bash
```

コンテナを起動すると次のエラーが発生。

> ModuleNotFoundError: No module named 'src'

気になってディレクトリ構造を確認してみたらADDしたファイルが作業フォルダに展開されていた。

```
$ docker exec -i -t a4883cee63b2 bash
bash: warning: setlocale: LC_ALL: cannot change locale (ja_JP.UTF-8)
root@a4883cee63b2:/app# ls
Pipfile       eda.ipynb         processed
Pipfile.lock  main.py           raw
__init__.py   modeling.py       utils.py
config.py     preprocessing.py
```

ADDやCOPYでフォルダのコピーを行う場合、`COPY コピーするフォルダ名（ローカル）/ ./コピーするフォルダ名（コンテナ）/` と書く。

```dockerfile
COPY Pipfile Pipfile.lock ./
COPY src/ ./src/
COPY data/ ./data/
```

### Dockerコンテナの稼働

`docker run -i -d -p 8080:[PORT番号] [イメージ名]` と打ち込むとコンテナポート8000を localhost 8080 にセットすることができる。

```
$ docker run -i -d -p 8080:8000 pipenv_docker
b2992d3b137f3f8229ab0ccc882533e85b2e851f4aadb489c0bb68c428b1563d
```

これで手元のブラウザから http://localhost:8080/ にアクセスすれば正常に動作しているのが確認できるはず。

参考

[Deploying Iris Classifications with FastAPI and Docker](https://towardsdatascience.com/deploying-iris-classifications-with-fastapi-and-docker-7c9b83fdec3a)


## Cloud Run でデプロイ

[クイックスタート: ビルドとデプロイ  |  Cloud Run  |  Google Cloud](https://cloud.google.com/run/docs/quickstarts/build-and-deploy?authuser=2)

### アプリをコンテナ化して Container Registry にアップロードする

`gcloud builds submit --tag gcr.io/[PROJECT-ID]/[PROJECT-NAME]`

```
gcloud builds submit CloudRun/south-park-dialogue --tag gcr.io/copywriting-helper/south-park-dialogue
```

### Cloud Run にデプロイする

`gcloud beta run deploy --image gcr.io/[PROJECT-ID]/[PROJECT-NAME]`

```
gcloud beta run deploy --image gcr.io/copywriting-helper/south-park-dialogue
```

途中でエラー発生。

```
   . Routing traffic...                            
  ✓ Setting IAM Policy...                         
Deployment failed                                 
ERROR: (gcloud.beta.run.deploy) Cloud Run error: Container failed to start. Failed to start and then listen on the port defined by the PORT environment variable. Logs for this revision might contain more information.
```

やり直しても駄目だった。
Cloudコンソールでログを確認すると「OpenBLAS WARNING - could not determine the L2 cache size on this system, assuming 256k」と表示されている。どうやらメモリ不足らしい。

[python - AppEngine warning - OpenBLAS WARNING - could not determine the L2 cache size on this system - Stack Overflow](https://stackoverflow.com/questions/55016899/appengine-warning-openblas-warning-could-not-determine-the-l2-cache-size-on)

```

```

```

```

```

```

```

```
