from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core.agent import ReActAgent
from app.tools import *
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables

app = FastAPI(title="LlamaIndex Agent API")

# Initialize Azure OpenAI LLM
llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the agent with tools
agent = ReActAgent.from_tools(
    tools=[summarize_text_tool, analyze_data_tool],
    llm=llm,
    verbose=True
)

class ChatInput(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "LlamaIndex Agent API is running"}

@app.post("/chat")
async def chat(input_data: ChatInput):
    try:
        response = agent.chat(input_data.message)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))