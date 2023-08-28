import os
import sys
from typing import List
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from sourced_conversation import ConversationalQAWithSourcesChain
# from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain

# Objectif : construire un chatbot qui permet d’utiliser les documents présent dans un dossier comme base de connaissance (ie : de discuter avec ses documents) en utilisant la technique de Retrieval Augmented Generation.

# Conseils :
# - Utilise Langchain comme backend
# - Pas besoin de faire un front, si on peut envoyer des messages dans une console Python, c’est suffisant
# - C’est bien si on peut savoir sur quels documents ont été utilisés pour répondre à la question
# - Tu peux installer un petit modèle sur ton laptop (les réponses seront pas dingue mais c’est pas grave) ou utiliser les crédits gratuits de l’API d'OpenAI

def get_loader(file: str) -> BaseLoader:
    """
    Return the loader corresponding to a given file. If the format is not supported, return None.
    Supported format : pdf, docx, doc, txt
    """
    path = "./source/" + file
    if file.endswith('.pdf'):
        return PyPDFLoader(path)
    elif file.endswith('.docx') or file.endswith('.doc'):
        return Docx2txtLoader(path)
    elif file.endswith('.txt'):
        return TextLoader(path)
    return None


def get_splitted_documents(folder_name: str, text_splitter: TextSplitter) -> List[str]:
    """
    Loads the documents in the given folder, and split them using the text_splitter.
    Returns a list of splitted documents.
    """
    loaders = [loader for loader in [get_loader(file) for file in os.listdir(folder_name)] if loader is not None]
    temp = [loader.load() for loader in loaders]
    temp = [doc for docs in temp for doc in docs]
    return text_splitter.split_documents(temp)

# We create a list that contains all the splitted documents loaded in the 'source' folder.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    separators=[" ", ",", "\n"]
)
documents = get_splitted_documents('source', text_splitter)

# Vector Database.
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="./data"
)

# Retriever used to load data if possible.
retriever = vectorstore.as_retriever(search_kwargs={'k': 7})

# Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

# Chain for Question-Answer discussion.
qa_chain = ConversationalQAWithSourcesChain.from_llm(
    OpenAI(temperature=0),
    retriever=retriever,
    return_source_documents=True,
    memory=memory
)

green = "\033[0;32m"
white = "\033[0;39m"
gray = "\033[0;90m"

while True:
    query = input(f"{green}You : ")
    if query == "stop" or query == "exit":
        print("End of conversation...")
        sys.exit()
    if query == "":
        continue
    answer = qa_chain({"question": query})
    print(f"{white}GPT : " + answer["answer"])
    print(f"{gray}Sources : " + answer["sources"])