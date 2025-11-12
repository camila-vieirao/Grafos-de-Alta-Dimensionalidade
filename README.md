# Trabalho de Grafos de Alta Dimensionalidade

Implementação de um sistema de análise de grafos em Python para a disciplina de Teoria dos Grafos.

## Requisitos Implementados

✅ Módulo de gravação em formato Pajek  
✅ Módulo de carregamento em formato Pajek  
✅ Verificação se o grafo é conexo  
✅ Identificação de componentes (fracamente conectados para direcionados)  
✅ Verificação se o grafo é Euleriano  
✅ Verificação se o grafo é Cíclico  
✅ Cálculo de Centralidade de Proximidade  
✅ Cálculo de Centralidade de Intermediação  
✅ Gerador de grafos aleatórios  
✅ Aplicação em dataset real (GitHub) com 5.000+ nós e 20.000+ arestas

## Estrutura do Projeto

```
src/
├── graph.py           # Classe principal do grafo
├── pajek.py           # Gerador de grafos aleatórios
├── github_dataset.py  # Análise do dataset GitHub
└── main.py            # Testes das funcionalidades
```

## Como Usar

### 1. Testar Funcionalidades Básicas

```bash
python src/main.py
```

### 2. Gerar Grafo Aleatório

```bash
python src/pajek.py
```

### 3. Analisar Dataset GitHub

Primeiro, baixe o dataset MUSAE GitHub ML e coloque os arquivos na raiz:
- `musae_git_target.csv`
- `musae_git_edges.csv`

Depois execute:

```bash
python src/github_dataset.py
```

## Exemplo de Uso da Classe Graph

```python
from graph import Graph

# Criar grafo
g = Graph(directed=False)
g.add_node("A")
g.add_node("B")
g.add_edge("A", "B", weight=5)

# Salvar em Pajek
g.save_pajek("meu_grafo.net")

# Carregar de Pajek
g2 = Graph.load_pajek("meu_grafo.net")

# Verificar propriedades
print("Conexo:", g2.is_connected())
print("Euleriano:", g2.is_eulerian())
print("Cíclico:", g2.is_cyclic())

# Calcular centralidades
closeness = g2.closeness_centrality()
betweenness = g2.betweenness_centrality()
```

## Dataset GitHub

O dataset utilizado representa a rede de desenvolvedores do GitHub:
- **Nós**: Usuários do GitHub
- **Arestas**: Relações de "seguir" entre usuários
- **Tamanho**: 5.000 usuários mais conectados, 20.000 conexões

### Resultados da Análise

- **Componentes**: 277 (1 componente principal com 4.724 nós)
- **Euleriano**: Não
- **Cíclico**: Não
- **Usuário mais central**: nfultz_0

## Formato Pajek

```
*Vertices 3
1 "Node A"
2 "Node B"
3 "Node C"
*Edges
1 2 5
2 3 3
```

## Dependências

- Python 3.7+
- Bibliotecas padrão (heapq, random, collections, csv)

Não requer instalação de bibliotecas externas.

