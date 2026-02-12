import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List

class VectorDatabaseManager:
    """Gerencia o banco de dados vetorial Chroma com Google Gemini Embeddings."""
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "models/gemini-embedding-001"):
        load_dotenv()
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    def create_or_update(self, documents: List[Document], force_recreate: bool = False):
        """Cria um novo banco ou carrega o existente."""
        exists = os.path.exists(self.persist_directory)
        if exists and not force_recreate:
            vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            return vectorstore, False # False indica que não foi criado do zero
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore, True # True indica que foi criado do zero

    def get_retriever(self, k: int = 5):
        """Retorna um objeto retriever para busca semântica."""
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Realiza busca semântica no banco de dados vetorial."""
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return vectorstore.similarity_search(query, k=k)

class RAGChainManager:
    """Gerencia o pipeline RAG com Reranking (Prompt + LLM + Retrieval)."""
    
    def __init__(self, vectorstore_manager: VectorDatabaseManager, model_name: str = "gemini-flash-latest"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.vectorstore_manager = vectorstore_manager
        
        # Template de Prompt Principal
        template = """
        Você é um assistente jurídico especializado em CDC e LGPD.
        Responda à pergunta do usuário utilizando APENAS o contexto fornecido abaixo.
        Se a resposta não estiver no contexto, diga que não possui informações suficientes.
        
        Ao responder, cite se a informação veio do "CDC" ou da "LGPD" com base nos metadados.
        
        Contexto:
        {context}
        
        Pergunta:
        {question}
        
        Resposta:
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Template de Reranking
        rerank_template = """
        Avalie a relevância do fragmento de texto abaixo para responder à pergunta fornecida.
        Atribua uma nota de 0 a 10, onde:
        0: Irrelevante
        10: Extremamente relevante e contém a resposta direta.
        
        Responda APENAS com o número (ex: 8.5).
        
        Pergunta: {question}
        Texto: {text}
        
        Nota:"""
        self.rerank_prompt = ChatPromptTemplate.from_template(rerank_template)

    def _format_docs(self, docs):
        formatted = []
        for doc in docs:
            fonte = doc.metadata.get("fonte", "desconhecida").upper()
            formatted.append(f"[{fonte}]: {doc.page_content}")
        return "\n\n".join(formatted)

    def rerank(self, question: str, docs: List[Document], k: int = 4) -> List[Document]:
        """Usa o LLM para reordenar docs por relevância em lote (Batch Reranking)."""
        if not docs:
            return []
            
        # Prepara o contexto com IDs
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"ID {i}: {doc.page_content}")
        
        batch_context = "\n\n".join(context_parts)
        
        template = """
        Abaixo estão vários fragmentos de texto (chunks) e uma pergunta. 
        Analise todos os fragmentos e selecione os {k} IDs dos fragmentos mais relevantes para responder à pergunta.
        Retorne APENAS os IDs separados por vírgula, em ordem de relevância (do mais relevante para o menos).
        Exemplo de retorno: 2, 5, 1, 0
        
        Pergunta: {question}
        
        Fragmentos:
        {context}
        
        IDs dos {k} mais relevantes:"""
        
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question, "context": batch_context, "k": k})
            
            # Limpa e extrai os IDs
            ids = [int(id_str.strip()) for id_str in response.split(",") if id_str.strip().isdigit()]
            
            # Retorna os documentos correspondentes (limitado a k e garantindo que os IDs são válidos)
            reranked_docs = []
            for idx in ids:
                if 0 <= idx < len(docs):
                    reranked_docs.append(docs[idx])
            
            return reranked_docs[:k] if reranked_docs else docs[:k]
        except Exception as e:
            print(f"Erro no reranking: {e}")
            return docs[:k]

    def ask(self, question: str, use_reranking: bool = True) -> str:
        """Processa uma pergunta via RAG, opcionalmente usando reranking."""
        if use_reranking:
            # 1. Recupera k=10 candidatos (reduzido de 15 para caber no contexto do prompt de reranking)
            initial_docs = self.vectorstore_manager.search(question, k=10)
            # 2. Rerank para pegar os Top 4
            final_docs = self.rerank(question, initial_docs, k=4)
        else:
            final_docs = self.vectorstore_manager.search(question, k=4)
            
        context = self._format_docs(final_docs)
        
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
