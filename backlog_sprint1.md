# Backlog da Sprint 1 | 02/02 a 13/02

Nesta Sprint, você deverá finalizar o curso "Arquiteturas RAG com LLMs: embeddings, busca semântica e criação de agentes com LangChain", em seguida, realizar a Lista de Exercícios 1 para fixação do conteúdo.

Encontro ao vivo 12/02 - quinta-feira - das 9h às 10h

Observação: Não é necessário entregar a resolução do exercício. Os exercícios práticos possuem o objetivo de que você coloque em prática o que aprendeu no curso da sprint como algo extra e complementar.

## 1 - Setup do projeto

### Descrição

🧩 ATIVIDADE PRÁTICA — RAG com Código de Defesa do Consumidor e LGPD

### Contexto

Você vai construir um assistente jurídico baseado em RAG capaz de responder perguntas sobre:

* Código de Defesa do Consumidor (CDC)
* Lei Geral de Proteção de Dados (LGPD)

### Preparação

No seu projeto, crie uma pasta chamada:

`rag-juridico/`

Dentro dela, organize:

```text
rag-juridico/
 ├── dados/
 │    ├── cdc.pdf
 │    └── lgpd.pdf
 ├── ingestao.py
 ├── rag.py
 └── app.py
```

Garanta que seu ambiente tenha:

* loader de PDF funcionando
* embeddings configurados
* Chroma

## 2 - Carregando documentos jurídicos

### Objetivo - Carregamento

Carregar os PDFs do CDC e da LGPD e transformá-los em documentos processáveis pelo LangChain.

### Tarefas - Carregamento

* Carregue os dois PDFs.
* Para cada página, adicione metadados:
  * fonte: "cdc" ou "lgpd"
* Ao final, imprima:
  * quantidade total de documentos carregados
  * quantidade por fonte (CDC vs LGPD)

### Resultado esperado - Carregamento

* Uma lista única de Document
* Metadados corretamente preenchidos

## 3 - Realizando chunking

### Objetivo - Chunking

Entender como a estratégia de chunking impacta o RAG.

### Tarefas - Chunking

* Crie duas funções de chunking:
  * Uma usando RecursiveCharacterTextSplitter
  * Outra quebrando por parágrafo (CharacterTextSplitter → \n\n)
* Gere chunks com:
  * tamanho fixo ≈ 500–800 caracteres
  * overlap configurável
* Compare:
  * número total de chunks gerados
  * tamanho médio dos chunks

### Pergunta para reflexão

Qual estratégia gera chunks mais “legíveis” para um texto jurídico?

## 4 - Criando embeddings e o banco vetorial

### Objetivo - Embeddings

Transformar chunks jurídicos em embeddings e armazená-los.

### Tarefas - Embeddings

* Gere embeddings para todos os chunks.
* Armazene em um vectorstore persistente.
* O banco deve permitir:
  * recarregar se já existir
  * criar do zero se não existir

### Extra (opcional)

* Crie coleções separadas:
  * uma para CDC
  * outra para LGPD

## 5 - Recuperação semântica

### Objetivo - Recuperação

Testar a busca vetorial antes de envolver o LLM.

### Tarefas - Recuperação

* Crie um retriever com k = 5.
* Faça buscas para perguntas como:
  * “O fornecedor pode se eximir de responsabilidade?”
  * “Em que casos o consentimento é obrigatório?”
* Exiba apenas:
  * o texto dos chunks recuperados
  * seus metadados

### Reflexão - Recuperação

Os chunks recuperados pertencem à lei correta?

## 6 - Primeira versão de RAG

### Objetivo - RAG

Integrar retrieval + LLM.

### Tarefas - RAG

* Monte um prompt que diga explicitamente:
  * “Responda somente com base no contexto fornecido.”
* Use os chunks recuperados como contexto.
* Responda perguntas como:
  * “O consumidor pode desistir da compra feita pela internet?”
  * “Quais são os direitos do titular de dados pessoais?”

### Resultado esperado - RAG

* Respostas corretas
* Fontes exibidas (CDC ou LGPD)

### Reflexão - RAG

E se a pergunta não estiver relacionada ao CDC ou LGPD? O modelo está preparado para isso?

## 7 - Reranking com LLM

### Objetivo - Reranking

Melhorar a qualidade do contexto usado pelo LLM.

### Tarefas - Reranking

* Recupere k = 15 chunks inicialmente.
* Use o LLM para dar uma nota de relevância (0 a 10) para cada chunk.
* Selecione apenas os 4 melhores.
* Use esses 4 no prompt final.

### Comparação

* Responda a mesma pergunta:
  * com reranking
  * sem reranking
* Compare clareza e precisão.
