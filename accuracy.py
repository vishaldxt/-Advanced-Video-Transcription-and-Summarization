from fastapi import FastAPI, HTTPException
from typing import Optional

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import os
os.environ['OPENAI_API_KEY'] = "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"

load_dotenv()

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """
    Load all the necessary models and data once the server starts.
    """
    app.directory = '/app/content/'
    app.documents = load_docs(app.directory)
    app.docs = split_docs(app.documents)

    app.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    app.persist_directory = "chroma_db"

    app.vectordb = Chroma.from_documents(
        documents=app.docs,
        embedding=app.embeddings,
        persist_directory=app.persist_directory
    )
    app.vectordb.persist()

    app.model_name = "gpt-3.5-turbo"
    app.llm = ChatOpenAI(model_name=app.model_name)
    app.llm = ChatOpenAI(openai_api_key="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")

    app.db = Chroma.from_documents(app.docs, app.embeddings)
    app.chain = load_qa_chain(app.llm, chain_type="stuff", verbose=True)


@app.get("/query/{question}")
async def query_chain(question: str):
    """
    Queries the model with a given question and returns the answer.
    """
    matching_docs_score = app.db.similarity_search_with_score(question)
    if len(matching_docs_score) == 0:
        raise HTTPException(status_code=404, detail="No matching documents found")

    matching_docs = [doc for doc, score in matching_docs_score]
    answer = app.chain.run(input_documents=matching_docs, question=question)

    # Prepare the sources
    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]

    return {"answer": answer, "sources": sources}


def load_docs(directory: str):
    """
    Load documents from the given directory.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()

    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    return docs

import matplotlib.pyplot as plt

def evaluate_accuracy(test_set, model):
    correct_answers = 0

    for question, expected_answer in test_set.items():
        generated_answer = model.query_chain(question)
        if is_correct(generated_answer, expected_answer):
            correct_answers += 1
    
    accuracy = correct_answers / len(test_set)
    return accuracy

def is_correct(generated_answer, expected_answer):
    # Define your criteria for a correct answer here
    return generated_answer.strip().lower() == expected_answer.strip().lower()

def plot_accuracy(accuracy):
    plt.figure(figsize=(5,5))
    plt.bar(['Accuracy'], [accuracy])
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.show()

# Assuming you have a 'test_set' dictionary where keys are questions and values are expected answers
accuracy = evaluate_accuracy(test_set, your_fastapi_app_instance)
plot_accuracy(accuracy)
