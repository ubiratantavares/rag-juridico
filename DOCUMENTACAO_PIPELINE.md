# Documentação Técnica: Pipeline RAG Jurídico

Este documento detalha o fluxo completo do projeto, desde o carregamento dos documentos legais até a geração de respostas especializadas utilizando Inteligência Artificial (RAG - *Retrieval-Augmented Generation*).

## 🏗️ Arquitetura Geral

O projeto utiliza uma arquitetura **Model-View-Controller (MVC)** adaptada para fluxos de dados de IA:

- **Model**: Gerenciado por `src/ingestao.py` (dados) e `src/rag.py` (IA).
- **View**: Implementada em `app.py` como a interface de terminal.
- **Controller**: Orquestra o fluxo entre o carregamento e a interação no `app.py`.

## 🚀 Fluxo Passo a Passo

### 1. Configuração e Ambiente

O pipeline começa garantindo que as credenciais e dependências estejam corretas.

- **Arquivo**: `.env`, `requirements.txt`.
- **Ação**: O script `tests/test_setup.py` valida se a `GOOGLE_API_KEY` existe e se as bibliotecas como `langchain-google-genai` e `chromadb` estão prontas.

### 2. Ingestão de Documentos (Ingestion)

O objetivo é transformar arquivos PDF brutos em objetos de dados que a IA compreenda.

- **Módulo**: `src/ingestao.py` (`LegalPDFLoader`).
- **Passo**:
  - O sistema lê os arquivos em `./dados/` (CDC e LGPD).
  - **Metadados**: Cada página recebe uma tag (`fonte: cdc` ou `fonte: lgpd`). Isso é crucial para que o modelo cite a fonte correta na resposta final.

### 3. Processamento de Texto (Chunking)

Documentos jurídicos são longos demais para o contexto do modelo. Precisamos "fatiá-los".

- **Módulo**: `src/ingestao.py` (`DocumentProcessor`).
- **Estratégia**: Utilizamos a `RecursiveCharacterTextSplitter` com:
  - **Chunk Size**: 1500 caracteres (tamanho ideal para manter o parágrafo).
  - **Overlap**: 300 caracteres (evita que uma frase seja cortada ao meio entre dois pedaços de papel).

### 4. Indexação Vetorial (Vector Store)

Transformamos os textos em números (vetores) para permitir buscas por significado.

- **Módulo**: `src/rag.py` (`VectorDatabaseManager`).
- **Embeddings**: Utilizamos o modelo `gemini-embedding-001`.
- **Banco**: O **ChromaDB** armazena esses vetores localmente na pasta `chroma_db/`. Isso permite que o sistema funcione sem precisar reprocessar os PDFs toda vez.

### 5. Recuperação Semântica (Retrieval)

Quando você faz uma pergunta, o sistema não busca por palavras exatas, mas por conceitos.

- **Módulo**: `src/rag.py` (`VectorDatabaseManager.search`).
- **Ação**: O sistema converte sua pergunta em um vetor e busca os fragmentos mais similares no ChromaDB.

### 6. Reranking com LLM

Para garantir a máxima precisão, incluímos uma etapa de refinamento.

- **Módulo**: `src/rag.py` (`RAGChainManager.rerank`).
- **Fluxo**:
    1. O sistema recupera 10 fragmentos candidatos (Busca Vetorial).
    2. Envia esses 10 pedaços para o **Gemini 2.0 Flash** em lote (*Batch*).
    3. O modelo avalia a relevância de cada um e devolve os **4 IDs mais importantes**.
    4. Esta técnica garante que a resposta final use apenas o contexto mais pertinente.

### 7. Geração de Resposta (Generation)

A fase final onde a resposta é redigida.

- **Módulo**: `src/rag.py` (`RAGChainManager.ask`).
- **Prompt**:
  - Instruímos o modelo a ser um "Assistente Jurídico".
  - Ele é proibido de usar conhecimento externo: **"Responda APENAS com o contexto fornecido"**.
  - Ele deve citar obrigatoriamente se a informação veio do CDC ou da LGPD.

### 8. Validação e Testes

Garantimos que cada engrenagem do pipeline continue funcionando.

- **Pasta**: `tests/`.
- **Ferramenta**: `pytest`.
- **Testes**: Cobrem desde o setup básico até a lógica complexa de reranking e a precisão da resposta do RAG.

## 🛠️ Como o Pipeline é Acionado

O usuário interage via `app.py`:

1. **Opção [1]**: Roda os passos 1 ao 4 (Limpa o banco e recria a partir dos PDFs).
2. **Opção [2]**: Roda os passos 5 ao 7 (Inicia o chat interativo com busca, rerank e resposta).

*Documentação gerada para a Sprint 1 do projeto RAG Jurídico.*
