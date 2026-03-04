"""
Enterprise RAG Pipeline - Core Module
ChromaDB + LangChain + HuggingFace Embeddings
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


class EnterpriseRAGPipeline:
    """
    Production-grade RAG pipeline with ChromaDB vector store,
    HuggingFace embeddings, and sliding window chunking strategy.
    """

    def __init__(self, use_openai: bool = False):
        logger.info("Initializing Enterprise RAG Pipeline...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None
        self.use_openai = use_openai
        self._init_llm()
        self._load_existing_db()

    def _init_llm(self):
        """Initialize LLM — supports OpenAI or local Ollama."""
        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info("Using OpenAI GPT-3.5-turbo")
        else:
            self.llm = Ollama(model="mistral", temperature=0.1)
            logger.info("Using local Ollama (mistral)")

    def _load_existing_db(self):
        """Load persisted ChromaDB if it exists."""
        if Path(CHROMA_DB_PATH).exists():
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
            )
            self._build_qa_chain()
            count = self.vectorstore._collection.count()
            logger.info(f"Loaded existing ChromaDB with {count} chunks.")

    def ingest_documents(self, source_path: str) -> dict:
        """
        Ingest documents from a file or directory.
        Supports: PDF, TXT
        Uses sliding window chunking with overlap for better retrieval.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {source_path}")

        # Load documents
        if path.is_dir():
            loader = DirectoryLoader(
                str(path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
            )
        elif path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")

        # Sliding window chunking strategy
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source_file"] = path.name

        # Store in ChromaDB
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DB_PATH,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.persist()
        self._build_qa_chain()

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vectorstore_size": self.vectorstore._collection.count(),
        }

    def _build_qa_chain(self):
        """Build the RetrievalQA chain with custom prompt template."""
        prompt_template = """You are an expert AI assistant. Use the following context to answer the question accurately.
If the context does not contain enough information, say "I don't have enough context to answer this question accurately."

Context:
{context}

Question: {question}

Answer (be concise and precise):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

    def query(self, question: str) -> dict:
        """
        Run semantic search + LLM inference for a question.
        Returns answer + source document references.
        """
        if self.qa_chain is None:
            raise RuntimeError("No documents ingested yet. Call ingest_documents() first.")

        result = self.qa_chain.invoke({"query": question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "source": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "preview": doc.page_content[:200] + "...",
            })

        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "num_sources": len(sources),
        }

    def get_stats(self) -> dict:
        """Return vectorstore statistics."""
        if self.vectorstore is None:
            return {"status": "empty", "chunks": 0}
        return {
            "status": "ready",
            "chunks": self.vectorstore._collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
