import os
from abc import ABC, abstractmethod
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

class DocumentLoader(ABC):
    """Interface para carregadores de documentos."""
    @abstractmethod
    def load(self) -> List[Document]:
        pass

class LegalPDFLoader(DocumentLoader):
    """Carregador especializado para PDFs jurídicos com metadados de fonte."""
    def __init__(self, file_path: str, source_label: str):
        self.file_path = file_path
        self.source_label = source_label

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")
        
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["fonte"] = self.source_label
            
        return docs

class DocumentProcessor:
    """Classe responsável pelo processamento (chunking) de documentos."""
    def __init__(self):
        pass

    def split_recursive(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    def split_by_paragraph(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 0) -> List[Document]:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

class IngestionManager:
    """Fachada para gerenciar o processo completo de ingestão."""
    def __init__(self, loaders: List[LegalPDFLoader]):
        self.loaders = loaders
        self.all_documents = []

    def load_all(self):
        """Carrega todos os documentos e retorna os documentos e estatísticas."""
        self.all_documents = []
        stats = {}
        for loader in self.loaders:
            loader_docs = loader.load()
            self.all_documents.extend(loader_docs)
            stats[loader.source_label] = len(loader_docs)
        
        stats["total"] = len(self.all_documents)
        return self.all_documents, stats
