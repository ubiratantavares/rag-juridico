import os
import pytest
from src.rag import VectorDatabaseManager
from langchain_core.documents import Document

# Este teste assume que a chave de API e o banco vetorial (mesmo que pequeno) estão configurados.
# Para evitar erros de cota do Gemini durante os testes, podemos mockar o retorno se necessário,
# mas aqui focaremos na integração conforme solicitado no backlog.

@pytest.fixture
def db_manager():
    return VectorDatabaseManager()

def test_retriever_creation(db_manager):
    """Valida se o retriever é criado com o parâmetro k correto."""
    retriever = db_manager.get_retriever(k=5)
    assert retriever.search_kwargs["k"] == 5

def test_semantic_search_integration(db_manager):
    """
    Testa a busca semântica com uma pergunta real.
    Nota: Este teste pode falhar se o banco estiver vazio ou se houver erro de cota.
    """
    if not os.path.exists("./chroma_db"):
        pytest.skip("Banco vetorial não encontrado. Execute a ingestão antes do teste.")
        
    query = "O fornecedor pode se eximir de responsabilidade?"
    results = db_manager.search(query, k=5)
    
    assert isinstance(results, list)
    assert len(results) > 0
    # Valida se os resultados contêm metadados esperados
    for doc in results:
        assert "fonte" in doc.metadata
        assert isinstance(doc.page_content, str)

def test_source_attribution(db_manager):
    """Valida se a busca retorna documentos das fontes corretas (CDC ou LGPD)."""
    if not os.path.exists("./chroma_db"):
        pytest.skip("Banco vetorial não encontrado.")
        
    # Pergunta sobre consentimento (espera-se LGPD)
    query = "Em que casos o consentimento é obrigatório?"
    results = db_manager.search(query, k=5)
    
    # Verifica se pelo menos um resultado menciona uma das fontes conhecidas
    fontes = [doc.metadata.get("fonte") for doc in results]
    assert any(f in ["cdc", "lgpd"] for f in fontes)
