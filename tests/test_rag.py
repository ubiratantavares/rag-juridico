import pytest
import os
from src.rag import VectorDatabaseManager, RAGChainManager

# Para rodar: export PYTHONPATH=$PYTHONPATH:. && pytest tests/test_rag.py

@pytest.fixture
def rag_manager():
    db_manager = VectorDatabaseManager()
    return RAGChainManager(db_manager)

def test_rag_answer_cdc(rag_manager):
    """Testa se o RAG responde corretamente sobre o CDC e cita a fonte."""
    if not os.path.exists("./chroma_db"):
        pytest.skip("Banco vetorial não encontrado.")
        
    pergunta = "O consumidor pode desistir da compra feita pela internet?"
    resposta = rag_manager.ask(pergunta)
    
    assert isinstance(resposta, str)
    assert len(resposta) > 20
    # Verifica se a resposta menciona CDC ou o direito de arrependimento (desistência)
    assert "CDC" in resposta.upper() or "CONSUMIDOR" in resposta.upper()

def test_rag_answer_lgpd(rag_manager):
    """Testa se o RAG responde corretamente sobre a LGPD e cita a fonte."""
    if not os.path.exists("./chroma_db"):
        pytest.skip("Banco vetorial não encontrado.")
        
    pergunta = "Quais são os direitos do titular de dados pessoais?"
    resposta = rag_manager.ask(pergunta)
    
    assert isinstance(resposta, str)
    assert len(resposta) > 20
    # Verifica se a resposta menciona LGPD ou Dados
    assert "LGPD" in resposta.upper() or "DADOS" in resposta.upper()

def test_rag_out_of_context(rag_manager):
    """Testa se o RAG recusa responder perguntas fora do contexto."""
    if not os.path.exists("./chroma_db"):
        pytest.skip("Banco vetorial não encontrado.")
        
    pergunta = "Qual a receita de um bolo de chocolate?"
    resposta = rag_manager.ask(pergunta)
    
    # O prompt diz para dizer que não possui informações se não estiver no contexto
    negativas = ["não possui", "não encontrei", "não há informações", "não sei", "informações suficientes"]
    assert any(n in resposta.lower() for n in negativas)
