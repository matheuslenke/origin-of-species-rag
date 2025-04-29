# Case de IA Generativa com RAG (Retrieval-Augmented Generation)

A ideia deste case é desenvolver uma aplicação RAG para responder questões precisas e contextualizadas baseadas no livro `A Origem das Espécies`

Este projeto utiliza a biblioteca `streamlit` que facilita a criação de uma interface de chat com scripts python customizados. Para o caso de uma aplicação real, provavelmente outra abordagem seria selecionada como o desenvolvimento de uma API em python com um frontend separado ou permitindo múltiplos frontends chamarem a mesma.

## Dependências técnicas

* `Python` versão 3.12
* `Poetry` para gerenciamento de dependências. Instalação [neste link](https://python-poetry.org/docs/#installation)
* `streamlit` para gerar de maneira simples uma interface de chat provisória
* `ruff` para linting de código
* `langchain` para utilização de LLMs
* `chromadb` para banco de dados de vetor


## Ambiente

Foi escolhido o `Poetry` para o gerenciamento do ambiente também, mas o `pyenv` pode ser utilizado também. A versão do python recomendada é a `3.12` ou superior.

Você pode ativar a versão necessária para este projeto com o comando:

```bash
poetry env use 3.12
```

Mais informações sobre ambientes do poetry podem ser encontradas [neste link](https://python-poetry.org/docs/managing-environments/)

## Comandos essenciais

No arquivo `pyproject.toml`, temos definidos alguns scripts para rodar a aplicação, contando com scripts para linting, build e execução.

1. Linting

```bash
poetry run poe lint
poetry run poe lint-fix
```

2. Build

```bash
poetry build
```


## Melhorias Futuras

* Migrar do langchain_google_genai para langchain_google_vertexai que permite limites maiores para aplicações comerciais