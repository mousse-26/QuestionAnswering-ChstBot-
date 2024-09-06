import locale
import asyncio
import getpass
import os
import certifi
import chainlit as cl
import aiohttp
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

os.environ['SSL_CERT_FILE'] = certifi.where()

base_dir = os.path.expanduser("~/PycharmProjects/HELLO/pythonProject")
pdf_files = [
    os.path.join(base_dir, "pdf", "HP 4.pdf")
]

# Load and split the PDF documents
pages = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages.extend(loader.load())

print(f"Loaded {len(pages)} pages from the PDF.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)

print(f"Split into {len(docs)} document chunks.")

# Create embeddings and a document search index
embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)

hf_token = getpass.getpass('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')

os.environ['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'] = hf_token

# Hugging Face model and its parameters
repo_id = "tiiuae/falcon-7b"
temperature = 0.5
llm = HuggingFaceEndpoint(
    endpoint_url=f"https://api-inference.huggingface.co/models/{repo_id}",
    api_token=hf_token,
    temperature=temperature
)

#RetrievalQA chain
@cl.on_chat_start
def on_chat_start():
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    cl.user_session.set("retrieval_chain", retrieval_chain)

# Handle incoming messages and generate responses based on the PDF content
@cl.on_message
async def on_message(message: cl.Message):
    retrieval_chain = cl.user_session.get("retrieval_chain")
    query = message.content

    attempt = 0
    max_attempts = 5
    initial_wait_time = 2

    while attempt < max_attempts:
        try:
            # Invoke the retrieval chain and send the response
            res = await retrieval_chain.ainvoke(query)
            answer = res["result"].strip()

            # Process to ensure only the simple answer is returned
            simple_answer = answer.split('\n')[0].strip()
            print(f"Query: {query}\nSimple Answer: {simple_answer}")  # Debug output for verification
            await cl.Message(content=simple_answer).send()
            break
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                # Handle rate limit errors with exponential backoff
                attempt += 1
                wait_time = initial_wait_time * (2 ** (attempt - 1))
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await cl.Message(content=f"Rate limit exceeded. Retrying in {wait_time} seconds...").send()
                await asyncio.sleep(wait_time)
            else:
                # Handle other client response errors
                print(f"ClientResponseError: {e.status}")
                await cl.Message(content=f"An error occurred: {e.status}").send()
                raise
        except Exception as e:
            # Handle unexpected errors
            print(f"An unexpected error occurred: {str(e)}")
            await cl.Message(content=f"An unexpected error occurred: {str(e)}").send()
            raise

if __name__ == "__main__":
    cl.main()
