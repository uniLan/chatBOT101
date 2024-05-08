from typing import Dict

import dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

OPENAI_API_KEY = ''
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key=OPENAI_API_KEY)
chat_history_data = ChatMessageHistory()
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""


def __init__():
    # temp_msge = dotenv.load_dotenv()

    # print(temp_msge)
    pass


def summarize_messages(chain_input):
    stored_messages = chat_history_data.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as "
                "you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | chat

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    chat_history_data.clear()

    chat_history_data.add_message(summary_message)

    return True


def trim_messages(chain_input):
    stored_messages = chat_history_data.messages
    if len(stored_messages) <= 2:
        return False

    chat_history_data.clear()

    for message in stored_messages[-2:]:
        chat_history_data.add_message(message)

    return True


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


def main():
    __init__()

    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

    retriever = vectorstore.as_retriever(k=4)  # initial retriever
    docs = retriever.invoke("can LangSmith help test my LLM app?")
    print(docs)
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    resp = document_chain.invoke(
        {
            "context": docs,
            "messages": [
                HumanMessage(content="Can LangSmith help test my LLM applications?")
            ],
        }
    )
    print(resp)

    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain,
    )

    resp = retrieval_chain.invoke(
        {
            "messages": [
                HumanMessage(content="Can LangSmith help test my LLM applications??")
            ],
        }
    )

    print(resp)

    resp = retriever.invoke("tell me more!")
    print(resp)


if __name__ == '__main__':
    main()
