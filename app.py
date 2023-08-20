import json
import os, csv
import requests
import pandas as pd
import concurrent.futures
from langchain.llms import OpenAI
from flask_cors import CORS, cross_origin
from langchain.vectorstores import Milvus
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, render_template, request, redirect


embedding = HuggingFaceEmbeddings()


app = Flask(__name__, template_folder='templates')

CORS(app, supports_credentials=True)

openai_api_key = "None"
openai_api_base = "http://172.29.7.155:8000/v1"
milvus_host = "172.29.4.47"
milvus_port = "19530"

llm_completion = OpenAI(model_name="vicuna-13b-v1.5")
llm_chat = ChatOpenAI(model_name="vicuna-13b-v1.5")
milvus = Milvus(
    embedding_function=embedding, 
    collection_name="arXiv_prompt", 
    connection_args={
        "host": milvus_host, 
        "port": milvus_port
    }
)
filename = 'history.csv'

# Check if history.csv file exists, otherwise create an empty file
if not os.path.isfile(filename):
    with open(filename, 'w', newline='') as file:
        fieldnames = ['query', 'answer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write column names to the file

# Global variable to store search history
search_history = pd.read_csv(filename).to_dict(orient='records')


def save_search_history():
    df = pd.DataFrame(search_history, columns=['query', 'answer'])
    df.to_csv(filename, index=False)


def askLLM(query_text: str):
    headers = {"Content-Type": "application/json"}
    parms = {
        "model": "vicuna-13b-v1.5",
        "messages": [{"role": "user", "content": query_text}],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stop": None,
        "n": 1,
        "top_p": 1.0,
    }
    return requests.post("http://172.29.7.155:8000/v1/chat/completions", headers=headers, json=parms).text


@app.route('/')
@cross_origin(supports_credentials=True)
def index():
    return render_template('index.html', search_history=search_history)


@app.route('/delete', methods=['POST'])
@cross_origin(supports_credentials=True)
def delete():
    global search_history
    search_history = []
    save_search_history()
    render_template('index.html', search_history=[])
    return redirect('/')


@app.route('/query', methods=['POST'])
@cross_origin(supports_credentials=True)
def query():
    query_text = request.form['query']

    # Store the search history in the CSV file
    save_search_history()

    # Few-shot Prompts
    prompt = f"""
        Q: give me the very scientific and professional words in the sentense \"\"\"What is the application of NLP\"\"\" in the form of a list without literally any explanation. if the sentense is trivial and not scientific, simply return me []
        A: ['NLP', 'application']
        Q: give me the very scientific and professional words in the sentense \"\"\"why is 1 but not 2?\"\"\" in the form of a list without literally any explanation. if the sentense is trivial and not scientific, simply return me []
        A: []
        Q: give me the very scientific and professional words in the sentense \"\"\"How is the development of formal methods\"\"\" in the form of a list without literally any explanation. if the sentense is trivial and not scientific, simply return me []
        A: ['formal', 'methods', 'development']
        Q: give me the very scientific and professional words in the sentense \"\"\"what about an ice cream\"\"\" in the form of a list without literally any explanation. if the sentense is trivial and not scientific, simply return me []
        A: []
        Q: give me the very scientific and professional words in the sentense \"\"\"{query_text}\"\"\" in the form of a list without literally any explanation. if the sentense is trivial and not scientific, simply return me []
        A: 
    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交并发任务
        future1 = executor.submit(askLLM, query_text=query_text)
        future2 = executor.submit(askLLM, query_text=prompt)
        # 获取结果
        explanation = future1.result()
        key_words = future2.result()

    # Process the explanation and retrieve relevant articles from Milvus
    explanation = json.loads(explanation)["choices"][0]["message"]["content"]
    key_words = json.loads(key_words)["choices"][0]["message"]["content"]
    print(key_words)
    articles = milvus.search(key_words, top_k=5, search_type="similarity") if len(key_words) > 2 else []
    articles = [
        {
            'title': article.metadata['title'],
            'authors': article.metadata['authors'],
            'url': 'https://arxiv.org/abs/' + article.metadata['access_id'],
            'abstract': article.page_content.strip(), 
        } for article in articles
    ]
    prompt = f"""
        I would give you a paragraph and a list of papers in a same topic.
        You are supposed to write a paragraph to integrate all the information I give you.  
        Just directly and fluently write the paragraph and explain the metadata respctively in natural language 
        (always remember to extract "titles", "authors" and "urls" like https://arxiv.org/abs/access_id from the list), 
        Here is the paragraph you should summary:
        \"\"\"{explanation}\"\"\"
        Here is the list of title, authors, url and abstract you should tell me to details and smoothly:
        \"\"\"{articles}\"\"\"
        (You are suggested to summary them like "an article named {articles[0]['title'] if articles else '______'} (available at {articles[0]['url'] if articles else '______'}) written by {articles[0]['authors'] if articles else '______'} introduces ....")
    """
    if articles != []:
        answer = askLLM(prompt)
        answer = json.loads(answer)["choices"][0]["message"]["content"] if answer else ""
    else:
        answer = explanation    
    # Add the query to the search history
    search_history.append({'query': query_text, 'answer': answer})
    return render_template('index.html', answer=answer, search_history=search_history)


app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

if __name__ == '__main__':
    app.run()