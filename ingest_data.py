from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle


print("Loading data...")
loader = UnstructuredFileLoader("lawfaq.txt")
raw_documents = loader.load()


print("Splitting text...")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
)
documents = text_splitter.split_documents(raw_documents)


print("Creating vectorstore...")
openai_api_key='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'
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
