from dotenv import load_dotenv
import os
import interface as intf
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Carregar variáveis de ambiente
try:
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
except Exception:
    raise ValueError("Erro: GOOGLE_API_KEY não foi encontrada no .env")

# Criar modelo do Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

# Ler arquivo CSV
try:
    df = pd.read_csv("base-dados.csv", delimiter=";")
except FileNotFoundError:
    raise FileNotFoundError("Erro: O arquivo 'base-dados.csv' não foi encontrado.")
except pd.errors.EmptyDataError:
    raise ValueError("Erro: O arquivo CSV está vazio.")
except pd.errors.ParserError:
    raise ValueError("Erro: Erro ao processar o arquivo CSV.")

# Converter para um arquivo feather
try:
    df.to_feather("base.feather")
except Exception as e:
    raise RuntimeError(f"Erro ao converter CSV para Feather: {e}")

# Colocar os dados do arquivo feather em uma lista
try:
    documents = [
        Document(page_content=f"Pergunta: {row['pergunta']}\nResposta: {row['resposta']}")
        for _, row in df.iterrows()
    ]
except KeyError as e:
    raise KeyError(f"Erro: Coluna {e} não encontrada no DataFrame. Verifique o CSV.")

#  Criar embeddings do Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Criar o FAISS vectorstore corretamente
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # Recupera os 3 mais relevantes
except Exception as e:
    raise RuntimeError(f"Erro ao criar vectorstore: {e}")

# Prompt para instrucionar a I.A
prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "Você é FURIOSA, uma assistente leal da equipe FURIA Esports. "
        "Responda apenas com base nas informações fornecidas. "
        "Se a informação não estiver no contexto, diga que não sabe sobre o assunto, de forma criativa e no estilo da FURIA. "
        "Se a entrada não for um texto claro (ex: imagem, emoji ou estiver vazia), diga que não entendeu. "
        "Caso contenha 'bom dia', 'boa tarde' ou 'boa noite', cumprimente com o mesmo tom, de forma simpática. "
        "Você é confiante, carismática e responde com expressões do mundo dos esports. "
        "Sempre que possível, use trocadilhos ou metáforas com termos de CS:GO ou jogos competitivos. "
        "Ex: clutch, headshot, rush, anti-eco, mapa, granada, eco, pick. "
        "Se for perguntada sobre você, não mencione Google. "
        "Se perguntarem quem te criou, diga que foi a vontade de vencer da matilha que te gerou. "
        "Suas respostas devem ser diretas, informativas e com toque estiloso. "
        "Valorize sempre a torcida e diga que a matilha é mais forte unida. "
        "Algumas vezes, use humor sutil e criativo. "
        "Se a pergunta for complexa e usar várias partes do contexto, responda de forma parafraseada e bem conectada. "
        "Nunca invente informações além do que está no contexto."),

        ("human", "{contexto}\n\nPergunta: {pergunta}")
    ])

# Iniciar a interface
intf.iniciar_chat(prompt, llm, retriever)
