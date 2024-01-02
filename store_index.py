from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

# print(PINECONE_API_ENV)
# print(PINECONE_API_KEY)

extracted_data = load_pdf('data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# pinecone.init(api_key=PINECONE_API_KEY,
#               env=PINECONE_API_ENV)
#
# index_name = pinecone.Index('medical-chatbot')
#
# docsearch = Pinecone.from_texts([t.page_content for t in text_chunks],
#                                 embedding=embeddings,
#                                 index_name=index_name)
# load it into FAISS
db = FAISS.from_documents(text_chunks, embeddings)