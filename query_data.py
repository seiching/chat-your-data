from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle

_template = """根據以下對話和一個隨後的問題，請將隨後的問題重述為一個獨立的問題。你可以假設這個問題是工程會採購法常用問題集.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """你是採購法常見問題集的客服專員及使用繁體中文(台灣)並不要用中國大陸PRC用語回答,如不要講軟件,u盤等
你要根據以下工程會採購法常用問題集和用戶的提問提供回答.
如果問題不在工程會採購法常用問題集內,例如常用問題集沒有提到公告金額是多少,你要回答,我不知道,並禮貌的說我只能以工程會採購法常見問題集回答.

Question: {question}
=========
{context}
=========

回答使用繁體中文(台灣)並以 Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])

import requests
from os import getcwd
chatmodlename="gpt-3.5-turbo"
def load_retriever():
    try:

      # url = "https://github.com/seiching/chat-your-data/raw/master/vectorstore.pkl"
       #directory = getcwd()
       #filename = directory + '/vectorstore.pkl'
       #print(filename)
       #r = requests.get(url)
       #f = open(filename,'w')
       #f.write(r.content)
       with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            print( 'file success')
    except:
       print( 'file error')
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_basic_qa_chain():
    llm = ChatOpenAI(model_name=chatmodlename, temperature=0,max_tokens=1000)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory)
    return model


def get_custom_prompt_qa_chain():
    llm = ChatOpenAI(model_name=chatmodlename, temperature=0,max_tokens=1000)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_condense_prompt_qa_chain():
    llm = ChatOpenAI(model_name=chatmodlename, temperature=0,max_tokens=1000)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_qa_with_sources_chain():
    llm = ChatOpenAI(model_name=chatmodlename, temperature=0,max_tokens=1000)
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question['question'], "chat_history": history}
        result = model(new_input)
        history.append((question['question'], result['answer']))
        return result

    return model_func


chain_options = {
    "basic": get_basic_qa_chain,
    "with_sources": get_qa_with_sources_chain,
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain
}
