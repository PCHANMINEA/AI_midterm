from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class UserInput(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "FastAPI is working!"}

@app.post("/generate")
def generate_text(data: UserInput):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": data.prompt}
        ]
    )
    answer = response.choices[0].message["content"]
    return {"response": answer}
