import pytest
import os
from src.rag import VectorDatabaseManager, RAGChainManager
from langchain_core.documents import Document

# Para rodar: export PYTHONPATH=$PYTHONPATH:. && pytest tests/test_reranking.py

@pytest.fixture
def rag_manager():
    db_manager = VectorDatabaseManager()
    return RAGChainManager(db_manager)

def test_rerank_logic_structure(rag_manager):
    """Valida se o método rerank retorna o número correto de documentos."""
    fake_docs = [
        Document(page_content=f"Conteúdo irrelevante {i}", metadata={"fonte": "cdc"})
        for i in range(10)
    ]
    # Adicionando um relevante
    fake_docs.append(Document(page_content="O consumidor tem direito ao arrependimento em 7 dias.", metadata={"fonte": "cdc"}))
    
    query = "Como funciona o direito de arrependimento?"
    top_docs = rag_manager.rerank(query, fake_docs, k=3)
    
    assert len(top_docs) == 3
    # O documento relevante deve estar entre os top (provavelmente o primeiro)
    assert any("7 dias" in doc.page_content for doc in top_docs)

def test_ask_with_reranking_integration(rag_manager):
    """Verifica se o método ask funciona com reranking ativado (integração)."""
    if not os.path.exists("./chroma_db"):
        pytest.skip("Banco vetorial não encontrado.")
        
    pergunta = "Quais as responsabilidades do fornecedor?"
    # Testa a chamada completa do pipeline com reranking
    resposta = rag_manager.ask(pergunta, use_reranking=True)
    
    assert isinstance(resposta, str)
    assert len(resposta) > 50
    assert "CDC" in resposta.upper()
