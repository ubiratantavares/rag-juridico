import os
import pytest
from dotenv import load_dotenv

# Para rodar: export PYTHONPATH=$PYTHONPATH:. && pytest tests/test_setup.py

def test_environment_variables():
    """Verifica se as chaves de API necessárias estão configuradas."""
    load_dotenv()
    assert os.getenv("GOOGLE_API_KEY") is not None, "GOOGLE_API_KEY não encontrada no .env"

def test_imports_gemini():
    """Verifica se as dependências do Google Gemini estão instaladas."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain_chroma import Chroma
    except ImportError as e:
        pytest.fail(f"Erro de importação de dependências essenciais: {e}")

def test_data_files_presence():
    """Verifica se os arquivos PDF jurídicos estão na pasta dados."""
    assert os.path.exists("./dados/cdc.pdf"), "Arquivo dados/cdc.pdf não encontrado"
    assert os.path.exists("./dados/lgpd.pdf"), "Arquivo dados/lgpd.pdf não encontrado"

def test_pdf_loader_functionality():
    """Valida se o carregador de PDF consegue abrir os arquivos."""
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader("./dados/cdc.pdf")
    docs = loader.load()
    assert len(docs) > 0, "O loader não conseguiu ler nenhuma página do CDC"

def test_embeddings_initialization():
    """Testa a inicialização dos embeddings do Gemini."""
    load_dotenv()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    # Tenta inicializar (pode falhar se a rede estiver offline, mas o teste foca na classe/env)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        assert embeddings is not None
    except Exception as e:
        pytest.fail(f"Falha ao inicializar GoogleGenerativeAIEmbeddings: {e}")
