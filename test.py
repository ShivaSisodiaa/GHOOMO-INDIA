# from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# import os
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# import langchain_openai

# from langchain_openai import OpenAI
# from langchain.chains import (
#     StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
# )
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import OpenAI
# import langchain_openai
# from langchain.agents.agent_types import AgentType
# import langchain_experimental.agents.agent_toolkits
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_openai import ChatOpenAI
# import pandas
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationSummaryMemory
# from langchain_openai import ChatOpenAI

import streamlit as st
import os

# os.environ["OPENAI_API_KEY"] = "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"

# embeddings = OpenAIEmbeddings()

# llm = ChatOpenAI()
# cwd = os.getcwd()

# directory = os.path.join(cwd,"moi_directory")

# def load_docs(directory):
#     loader = DirectoryLoader(directory)
#     documents = loader.load()
#     return documents

# documents = load_docs(directory)

# def split_docs(documents, chunk_size=500, chunk_overlap=50):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     docs = text_splitter.split_documents(documents)
#     return docs

# docs = split_docs(documents)

# db = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings
# )

# chain = load_qa_chain(llm, chain_type="stuff")

# def get_answer(query):
#     similar_docs = db.similarity_search(query, k=2) # get two closest chunks
#     answer = chain.run(input_documents=similar_docs, question=query)
#     return answer



# # Your custom imports and setup here...
# # ...

import streamlit as st
import os
from openai import OpenAI
st.title("GHOOMO INDIA")
os.environ["OPENAI_API_KEY"] = "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"
# Set up OpenAI client
openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Which corner of the world is calling your spirit of adventure?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Use ChatGPT to get the answer
        response = openai.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt}
          ]
        )
        response_message = response.choices[0].message.content
        st.write(response_message)
    st.session_state.messages.append({"role": "assistant", "content": response_message})
