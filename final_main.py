from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from deepspeech import Model
import numpy as np
import wave
import subprocess
from pytube import YouTube
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
os.environ['OPENAI_API_KEY'] = "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"

load_dotenv()

app = FastAPI()

# Assume that the models are in the root of the working directory
model_file_path = '/app/Models/deepspeech-0.9.3-models.pbmm'
scorer_file_path = '/app/Models/deepspeech-0.9.3-models.scorer'
content_directory = '/app/content/'
downloads_path = '/app/downloads'
transcripts_path = '/app/transcripts'
chroma_db_directory = "chroma_db"

# Load DeepSpeech model
model = Model(model_file_path)
model.enableExternalScorer(scorer_file_path)

# Initialize LangChain components
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(openai_api_key="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")


@app.on_event("startup")
async def startup_event():
    """
    Load all the necessary models and data once the server starts.
    """
    app.documents = load_docs(content_directory)
    app.docs = split_docs(app.documents)
    app.vectordb = Chroma.from_documents(
        documents=app.docs,
        embedding=embeddings,
        persist_directory=chroma_db_directory 
    )
    app.vectordb.persist()
    app.llm = llm
    app.chain = load_qa_chain(app.llm, chain_type="stuff", verbose=True)


@app.post("/download_and_transcribe/")
async def download_and_transcribe(youtube_url: str):
    """
    Downloads audio from a YouTube URL, transcribes it, and saves the transcript.
    """
    try:
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file_name = audio_stream.download(downloads_path)

        base, _ = os.path.splitext(audio_file_name)
        wav_file_path = base + '.wav'

        conversion_command = ['ffmpeg', '-y', '-i', audio_file_name, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wav_file_path]
        conversion_process = subprocess.run(conversion_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if conversion_process.returncode != 0:
            error_details = conversion_process.stderr.decode('utf-8')
            raise HTTPException(status_code=500, detail=f"Error converting file to WAV: {error_details}")

        # Perform speech recognition with DeepSpeech
        with wave.open(wav_file_path, 'rb') as audio_wave:
            audio_data = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), np.int16)
            text = model.stt(audio_data)

        # Save the transcript in the content directory instead of transcripts_path
        #transcript_file_name = os.path.basename(base) + '.txt'
        transcript_file_name = os.path.basename(base).replace(' ', '_') + '.txt'

        transcript_file_path = os.path.join(content_directory, transcript_file_name)  # Changed to content_directory
        with open(transcript_file_path, 'w') as file:
            file.write(text)

        # After saving the transcript, refresh documents for the QA syst em 
        await refresh_documents()

        return FileResponse(path=transcript_file_path, filename=transcript_file_name, media_type='text/plain')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh_documents/")
async def refresh_documents():
    """
    Refresh the documents from the content directory.
    """
    app.documents = load_docs(content_directory)
    app.docs = split_docs(app.documents)
    app.vectordb = Chroma.from_documents(
        documents=app.docs,
        embedding=embeddings,
        persist_directory=chroma_db_directory
    )
    app.vectordb.persist()
    app.db = app.vectordb
    app.chain = load_qa_chain(app.llm, chain_type="stuff", verbose=True)
    return {"message": "Documents refreshed and vector database updated."}


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

