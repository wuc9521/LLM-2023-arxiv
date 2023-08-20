## 1. Prerequisites
This project uses the LLM deployed in Nanjing University Software Institude. In case that you are not in the campus of NJU, please visit <https://vpn.nju.edu.cn>.


## 2. Run the project

```python
pip3 install -r ./requirements 
python3 -B ./app.py
```
and then open the browser and go to http://localhost:5000



## 3. Project demo

see [this link](https://box.nju.edu.cn/f/652e97d921f44007ad83/) for detailed demo video



## 4. Try Docker

```python
docker build -t arxiv-query-system .
docker run -p 5000:5000 arxiv-query-system
```

and then open the browser and go to http://localhost:5000