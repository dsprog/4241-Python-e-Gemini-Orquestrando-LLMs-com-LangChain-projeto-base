import openai
from my_keys import MARITACA_API_KEY

client = openai.OpenAI(
    api_key=MARITACA_API_KEY,
    base_url="https://chat.maritaca.ai/api",
)

response = client.chat.completions.create(
  model="sabia-3",
  messages=[
    {"role": "user", "content": "Qual é o significado da expressão 'chutar o balde' no Brasil?"},
  ],
  max_tokens=8000
)
answer = response.choices[0].message.content

print(f"Resposta: {answer}")  