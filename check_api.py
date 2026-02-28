import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

client=genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

question=input("enter question:\n")

response=client.model.generate_content(
    model="gemini-3-flash-preview",
    contents=question
)
print(response.text)