import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo

os.environ['OPENAI_API_KEY'] = ""

llm = OpenAI(temperature=0.9)
embedding = OpenAIEmbeddings()

st.title("Train GPT [WIP]")

loader = PyPDFLoader("train_gpt/howto-ipaddress.pdf")

pages = loader.load_and_split()

store = Chroma.from_documents(pages, embedding, collection_name="howto-ipaddress")

vectorstore_info = VectorStoreInfo(
    name = "howto-ipaddress",
    description = "Python 3.11.4 Documentation",
    vectorstore=store
    )

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,

)
prompt = st.text_input("Write your prompt: ")

if prompt:

    response = llm(prompt)
    
    response = agent_executor.run(prompt)

    st.write(response)

    with st.expander("Document Similarity Search"):

        search = store.similarity_search_with_score(prompt)

        st.write(search[0][0].page_content)

