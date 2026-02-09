import sys
import os

def test_imports():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        print("✅ Imports automáticos via LangChain: OK")
        return True
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False

def test_pdf_loader():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        pdf_path = "dados/cdc.pdf"
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            print(f"✅ Loader de PDF: OK (Carregadas {len(pages)} páginas de {pdf_path})")
            return True
        else:
            print(f"⚠️ Arquivo {pdf_path} não encontrado para teste de loader.")
            return False
    except Exception as e:
        print(f"❌ Erro no Loader de PDF: {e}")
        return False

def test_chroma():
    try:
        from langchain_chroma import Chroma
        # Teste de inicialização básica sem persistência real para evitar poluição
        # Usamos uma coleção em memória se possível, mas aqui testamos apenas a classe
        print("✅ Chroma (VectorStore): OK")
        return True
    except Exception as e:
        print(f"❌ Erro no Chroma: {e}")
        return False

if __name__ == "__main__":
    print(f"Iniciando verificação no ambiente: {sys.prefix}\n")
    s1 = test_imports()
    s2 = test_pdf_loader()
    s3 = test_chroma()
    
    if s1 and s2 and s3:
        print("\n🚀 Ambiente garantido e pronto para uso!")
    else:
        print("\n⚠️ O ambiente possui pendências. Verifique os erros acima.")
