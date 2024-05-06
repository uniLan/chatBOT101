import dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

OPENAI_API_KEY = ''
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key=OPENAI_API_KEY)
chat_history_data = ChatMessageHistory()


def __init__():
    temp_msge = dotenv.load_dotenv()

    print(temp_msge)


def summarize_messages(chain_input):
    stored_messages = chat_history_data.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
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


def main():
    __init__()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | chat

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history_data,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    chain_with_trimming = (
            RunnablePassthrough.assign(messages_trimmed=trim_messages)
            | chain_with_message_history
    )

    while True:
        user_input = input("Enter here: ")
        resp = chain_with_trimming.invoke(
            {"input": user_input},
            {"configurable": {"session_id": "unused"}},
        )
        print('BOT:', resp.content)


if __name__ == '__main__':
    main()
