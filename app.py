import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from docx import Document as DocxDocument
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.document_loaders import Docx2txtLoader
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

os.environ["AZURE_OPENAI_API_VERSION"]="2024-05-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]="text-embedding-ada"
os.environ["AZURE_OPENAI_ENDPOINT"]= "https://gen-ai-training-internal-july.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"]= "23ba1b747e164a38b3e537573b49cf60"

DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_documents(word_docs):
    documents = []
    
    for uploaded_file in word_docs:
        # Save the uploaded file to a temporary location
        temp_file_path = f"{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Use the path with Docx2txtLoader
        document_loader = Docx2txtLoader(temp_file_path)  # Use Docx2txtLoader for .docx files
        documents.extend(document_loader.load())
        return documents
    
    return documents
    
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Adjust chunk size
        chunk_overlap=50      # Adjust overlap size
    )
    return text_splitter.split_documents(documents)

# Split the documents into smaller chunks


# Inject custom CSS

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
st.set_page_config(page_title="Bank Smart Search")
custom_css = """
    <style>
        /* General body styling */
        body {
            font-family: "proxima_nova", Arial, sans-serif !important;
        }
        .response {
            color: black;
        }
        .st-emotion-cache-1c7y2kd {
            background-color: unset;
        }
        .st-emotion-cache-uzeiqp {
            font-family: "proxima_nova", Arial, sans-serif !important;
            color: #ec0100 !important;
        }
        .st-emotion-cache-187vdiz {
            font-family: "proxima_nova", Arial, sans-serif !important;
            color: #ec0100 !important;
        }
        .adbc-title {
            font-family: "proxima_nova", Arial, sans-serif !important;
            color: #ec0100 !important;
        }

        /* Title styling */
        .stApp h1 {
            color: #ec0100;
        }

        /* Sidebar button styling */
        .css-1l4w6pd { 
            font-size: 14px !important; 
        }

        /* Adjust the chat message styling */
        .css-1q8dd3e {
            font-size: 14px !important;
        }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)



st.markdown('<h1 class="adbc-title">Bank Smart Search</h1>', unsafe_allow_html=True)
st.write('This chatbot will provide the knowledge based on the bank provided document')

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


    
    
    


def load_llm():
    llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="GPT35",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
    return llm


    return llm.invoke(query).content


def retrieval_qa_chain(llm, prompt, db):
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       
                                       )
    return qa_chain


def qa_bot():
    """embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    
)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)"""
    llm = load_llm()
    qa_prompt = set_custom_prompt()

    qa = retrieval_qa_chain(llm, qa_prompt, st.session_state.db)

    return qa

def final_result(query):
   
    #query=format_prompt_with_history(st.session_state.messages,query)
    qa_result = qa_bot()
    response = qa_result({'query': query})
    
    return response


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Central Bank Of UAE ask anything related to cerdit facilities?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "assistant":
            st.write(message["content"])
        else:
            st.markdown(f'<div class="response">{message["content"]}</div>', unsafe_allow_html=True)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Central Bank Of UAE ask anything related to cerdit facilities?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)




if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer= final_result(prompt)
            response = answer['result']
           
            source_documents = answer['source_documents']
            print(source_documents)
                        
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                # st.write("case 2")
                # placeholder.markdown(full_response)
            # st.write("case 3")

            #pages = ''
            pagecontent = ''
            for doc in source_documents:
                
                #page_no = str(doc.metadata.get('page')+1)
                #pages  += page_no+","
                pagecontent += doc.page_content
                if "I'm sorry"  in response or "I don't know" in response:
                    placeholder.markdown(f'<div class="response">{full_response}</div>', unsafe_allow_html=True)
                else:    
                    placeholder.markdown(f'<div class="response">{full_response}</div><div class="response" Sources:></div>'
                                     f'<div class="response">Reference document-{source_documents[0].metadata.get('source')}</div>'
                                     f'<div class="response"> Chunk Content:. {pagecontent}</div>', unsafe_allow_html=True)
                                     #f'<div class="response"> Page No. {pages}</div>'
                # placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


with st.sidebar:
    st.subheader("Your documents")
    word_docs = st.file_uploader(
        "Upload your word docs here and click on 'Process'", accept_multiple_files=True)
    
    if st.button("Process"):
        with st.spinner("Processing"):
             doc=load_documents(word_docs) 
             chunks = split_documents(doc)
             
             embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],

            )

             st.session_state.db = FAISS.from_documents(chunks, embeddings)
                
