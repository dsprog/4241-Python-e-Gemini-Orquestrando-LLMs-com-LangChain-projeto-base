from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH, GEMINI_PRO, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

set_debug(True)

llm = ChatGoogleGenerativeAI(
  api_key=GEMINI_API_KEY,
  model=GEMINI_FLASH
)

# resposta = llm.invoke("Quais canais de Youtube você recomenda para que eu possa saber mais a respeito de smarpthones?")
# print("Gemini: ", resposta.content)

llm = ChatMaritalk(
  api_key=MARITACA_API_KEY,
  model=MARITACA_SABIA
)
# resposta = llm.invoke("Quais canais de Youtube você recomenda para que eu possa saber mais a respeito de smarpthones?")
# print("Maritaca: ", resposta.content)

imagem = encode_image("dados\exemplo_grafico.jpg")

template_analisador = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Assuma que você é um analisador de imagens. Sua tarefa é analisar a imagem
            e extrair informações de forma objetiva.

            # FORMATO DE SAÍDA
            Descrição da Imagem: 'Coloque a sua descrição da imagem aqui'
            Rótulos: 'Coloque uma lista com três termos chave separados por vírgula'
            """
        ),
        (
            "user",
            [
                {"type" : "text", "text" : "Descreva a imagem:"},
                {"type" : "image_url", "image_url" : {"url":"data:image/jpeg;base64,{imagem_informada}"}}
            ]
        )
    ]
)

cadeia_resumo = template_analisador | llm | StrOutputParser()
# resposta = cadeia_resumo.invoke({"imagem_informada" : imagem})

template_resposta = PromptTemplate(
    template="""
    Gere um resumo, utilizando uma linguagem clara e objetiva, focada no publico brasileiro.
    A ideia é que a comunicação do resultado seja mais fácil possivel, priorizando registros para consultas posteriores.
    
    # Resultado da imagem
    {resultado_imagem}
    """,
    input_variables=["resultado_imagem"]
)

llm_maritaca = ChatMaritalk(
  api_key=MARITACA_API_KEY,
  model=MARITACA_SABIA
)

cadeia_resumo = template_resposta | llm_maritaca | StrOutputParser()

cadeia_competa = (template_analisador | cadeia_resumo)

resposta = cadeia_competa.invoke({"imagem_informada" : imagem})

print(resposta)
