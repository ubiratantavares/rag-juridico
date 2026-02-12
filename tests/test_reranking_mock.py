import pytest
from unittest.mock import MagicMock, patch
from src.rag import RAGChainManager, VectorDatabaseManager
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

# Para rodar: export PYTHONPATH=$PYTHONPATH:. && pytest tests/test_reranking_mock.py

@pytest.fixture
def rag_manager():
    # Mocka a classe ChatGoogleGenerativeAI para evitar validação de API Key na inicialização
    with patch("src.rag.ChatGoogleGenerativeAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        db_manager = MagicMock(spec=VectorDatabaseManager)
        manager = RAGChainManager(db_manager)
        return manager

def test_rerank_logic_parsing_mocked(rag_manager):
    """Valida se o método rerank processa corretamente a resposta do LLM."""
    # Simula resposta do LLM usando um AIMessage real (ou compatível)
    rag_manager.llm.invoke.return_value = AIMessage(content="2, 0")
    
    # Documentos de teste
    docs = [
        Document(page_content="Doc 0", metadata={"fonte": "cdc"}),
        Document(page_content="Doc 1", metadata={"fonte": "cdc"}),
        Document(page_content="Doc 2", metadata={"fonte": "cdc"})
    ]
    
    # Executa o rerank pedindo Top 2
    reranked = rag_manager.rerank("Pergunta", docs, k=2)
    
    # Verifica se os docs retornados seguem a ordem do LLM (Doc 2, depois Doc 0)
    assert len(reranked) == 2
    assert reranked[0].page_content == "Doc 2"
    assert reranked[1].page_content == "Doc 0"

def test_rerank_fallback_on_error(rag_manager):
    """Valida se o rerank retorna os docs originais em caso de erro no LLM."""
    # Simula erro no LLM
    rag_manager.llm.invoke.side_effect = Exception("Quota exceeded")
    
    docs = [Document(page_content="Doc 0"), Document(page_content="Doc 1")]
    
    # Deve retornar os docs originais (fallback) em vez de quebrar
    reranked = rag_manager.rerank("Pergunta", docs, k=1)
    assert len(reranked) == 1
    assert reranked[0].page_content == "Doc 0"
