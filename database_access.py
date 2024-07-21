from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Create vector store if not exist
def create_vector_store(persistent_directory, documents, text_splitter, embeddings_model = "hkunlp/instructor-large"):
 
      # Split the document into chunks
      docs = text_splitter.split_documents(documents)
      
      #  Create embeddings model
      print("\n--- Creating embeddings ---")
      embeddings = HuggingFaceEmbeddings(model_name = embeddings_model) # Update to a valid embedding model if needed
      print("\n--- Finished creating embeddings ---")

      # Create the vector store and persist it automatically
      print("\n--- Creating vector store ---")
      db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
      print("\n--- Finished creating vector store ---")
      return db.as_retriever(search_kwargs={'k': 7})

# add document to the vector store
def add_vector(database_path, documents, text_splitter, embeddings_model = "hkunlp/instructor-large"):
      
      # Split the document into chunks
      docs = text_splitter.split_documents(documents)
      
      #  Create embeddings model
      print("\n--- Creating embeddings ---")
      embeddings = HuggingFaceEmbeddings(model_name = embeddings_model) # Update to a valid embedding model if needed
      print("\n--- Finished creating embeddings ---")

      # Load the existing database
      database = Chroma(persist_directory=database_path, embedding_function=embeddings)
      # Add the vector to the existing database
      print("--- Adding to database ---")
      database.add_documents(documents=docs)
      
      # save to the db
      database.persist()
      print("Finishing adding new documents")
      return database.as_retriever(search_kwargs={'k': 7})