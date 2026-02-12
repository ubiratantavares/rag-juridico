from typing import List, Dict
from src.ingestao import LegalPDFLoader, IngestionManager, DocumentProcessor
from src.rag import VectorDatabaseManager, RAGChainManager

class RAGView:
    """Responsável por toda a interface de saída para o usuário (Console)."""
    
    @staticmethod
    def exibir_titulo(titulo: str):
        print(f"\n{'='*10} {titulo} {'='*10}")

    @staticmethod
    def exibir_estatisticas_carregamento(stats: Dict):
        print("\nEstatísticas de Carregamento:")
        print(f"  - Total de documentos: {stats.get('total')}")
        for fonte, count in stats.items():
            if fonte != "total" and not fonte.startswith("total_"):
                print(f"  - Fonte {fonte.upper()}: {count}")

    @staticmethod
    def exibir_estatisticas_chunking(nome: str, total_chunks: int, tamanho_medio: float):
        print(f"\nEstratégia: {nome}")
        print(f"  - Total de chunks: {total_chunks}")
        print(f"  - Tamanho médio: {tamanho_medio:.2f} caracteres")

    @staticmethod
    def exibir_status_banco(criado_novo: bool, persist_directory: str, num_chunks: int = 0):
        if criado_novo:
            print(f"\nNovo banco vetorial criado em {persist_directory} com {num_chunks} chunks.")
        else:
            print(f"\nBanco vetorial carregado de {persist_directory}.")

    @staticmethod
    def exibir_sucesso(mensagem: str):
        print(f"\n{mensagem}")
        
    @staticmethod
    def exibir_resposta_rag(pergunta: str, resposta: str):
        print(f"\nPergunta: {pergunta}")
        print(f"Resposta: {resposta}")
        print("-" * 50)

class RAGController:
    """Orquestrador do Pipeline RAG (Padrão Controller)."""
    
    def __init__(self, view: RAGView):
        self.view = view
        self.processor = DocumentProcessor()
        self.db_manager = VectorDatabaseManager()
        self.rag_manager = RAGChainManager(self.db_manager)

    def executar_pipeline_ingestao(self):
        self.view.exibir_titulo("PIPELINE DE INGESTÃO RAG")
        
        # 1. Ingestão (Model)
        loaders = [
            LegalPDFLoader(file_path="./dados/cdc.pdf", source_label="cdc"),
            LegalPDFLoader(file_path="./dados/lgpd.pdf", source_label="lgpd")
        ]
        ingestion_manager = IngestionManager(loaders)
        documents, stats = ingestion_manager.load_all()
        
        # Limitando para evitar RESOURCE_EXHAUSTED no Gemini Free Tier
        documents = documents[:20]
        self.view.exibir_estatisticas_carregamento(stats)
        
        # 2. Chunking (Model)
        chunks_recursive = self.processor.split_recursive(documents, chunk_size=1500, chunk_overlap=300)
        tamanho_medio_rec = sum(len(c.page_content) for c in chunks_recursive) / len(chunks_recursive)
        self.view.exibir_estatisticas_chunking("RecursiveCharacterTextSplitter", len(chunks_recursive), tamanho_medio_rec)
        
        # 3. Vector Store (Model)
        vectorstore, criado_novo = self.db_manager.create_or_update(chunks_recursive)
        self.view.exibir_status_banco(criado_novo, self.db_manager.persist_directory, len(chunks_recursive))
        
        self.view.exibir_sucesso("Pipeline executado com sucesso!")

    def executar_chat(self):
        self.view.exibir_titulo("ASSISTENTE JURÍDICO (CDC & LGPD)")
        print("Digite sua pergunta ou 'sair' para encerrar.")
        
        while True:
            pergunta = input("\nVocê: ")
            if pergunta.lower() in ["sair", "exit", "quit"]:
                break
            
            resposta = self.rag_manager.ask(pergunta)
            self.view.exibir_resposta_rag(pergunta, resposta)

def main():
    view = RAGView()
    controller = RAGController(view)
    
    print("\n[1] Rodar Ingestão (reconstruir banco)")
    print("[2] Abrir Chat Assistente")
    opcao = input("\nEscolha uma opção: ")
    
    if opcao == "1":
        controller.executar_pipeline_ingestao()
    else:
        controller.executar_chat()

if __name__ == "__main__":
    main()
