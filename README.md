# Oráculo de Perguntas Utilizando IA
 Um oráculo de perguntas feito com o uso da IA Gemini fornecida pelo Google. A base de dados é extraída de um **arquivo.csv** contendo perguntas e respostas.
 
 A análise é feita utilizando a técnica de **embeddings** para criar valores de proximidade com **vetores tridimensionais**, que representam de forma eficiente as relações semânticas entre as perguntas e respostas na base de dados.  

# Funcionalidades
- **Chat com IA:** O usuário pode fazer perguntas de acordo com a base de dados e obter respostas geradas pela IA.
- **Base de Dados:** O sistema carrega perguntas e respostas relacionadas a equipe FURIA Esports a partir de um arquivo CSV, transformando-o em um Feather, utilizando Pandas, para aumentar a eficiência.
- **Modelo de IA:** Utiliza o modelo Google Gemini para fornecer respostas baseadas no contexto das perguntas.

# Técnologias Utilizadas
- Python
- Langchain
- Dotenv
- Google Gemini
- HuggingFace
- Streamlit
- CSV
- Pandas
- Feather

instale, caso não as possua:
```
pip install -r requirements.txt
```

# Explicação do Código
WARNING⚠️: Altere o modelo da IA Google Gemini de acordo com a sua conta!
```
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
```
O código está dividido em algumas partes principais:

## 1. Carregamento da Base de Dados
A base de dados é um arquivo CSV contendo perguntas e respostas relacionadas a FURIA. As perguntas são processadas e armazenadas para análise posterior.
## 2. Criação dos Embeddings
Usamos o modelo HuggingFace Embeddings para criar representações vetoriais das perguntas. Esses vetores são usados para medir a similaridade entre as perguntas feitas pelo usuário e as perguntas na base de dados.
## 3. Modelo de IA - Google Gemini
A IA utilizada para gerar as respostas é o Google Gemini, que é invocada para fornecer respostas com base no contexto das perguntas feitas. O modelo é alimentado com as informações recuperadas pela técnica de embeddings.
## 4. Interação com o Usuário via Streamlit
O Streamlit é usado para criar uma interface simples de chat, onde o usuário pode digitar perguntas e obter respostas em tempo real.
## 5. Estrutura de Consulta
O processo de consulta ao banco de dados é feito recuperando as 3 perguntas mais similares àquela realizada pelo usuário, concatenando suas respostas e usando essas informações para gerar uma resposta relevante.


## ⚠️ OBS: Antes de executar o programa, é necessário configurar sua chave da API do Google Gemini. Para isso, crie um arquivo .env no diretório do projeto e adicione a seguinte linha:
```
GOOGLE_API_KEY = sua_chave_api
```
# Como utilizar

Em seu terminal, acesse o diretório e digite o comando:

**Windows** (Powershell ou Prompt De Comando):  
```
python -m streamlit run oracle.py
```
**Linux ou macOS:  
```
streamlit run oracle.py
```
