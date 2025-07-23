import json
from llama_index.core.tools import FunctionTool
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from dotenv import load_dotenv

load_dotenv() 


llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)
def summarize_text(text: str) -> str:
    SUMMARY_TEXT = """
    You are an expert summarizing information about texts that you receive. Please provide a short summary about the data contained in the text you receive.

    The text is:
    {text}

    Answer:
    """

    PROMPT_SUMMARY = PromptTemplate(SUMMARY_TEXT)

    response = llm.complete(
        PROMPT_SUMMARY.format(
            text = text
        )
    )
    return f"Summary: {response.text}" 

summarize_text_tool = FunctionTool.from_defaults(
    fn=summarize_text,
    name="summarize_text",
    description="Summarizes a given text string into a concise summary."
)

def analyze_data(data: str) -> str:
    try:
        # Parse JSON input
        parsed_data = json.loads(data)
        
        ANALYSIS_TEXT = """
        You are an expert data analyst. Analyze the provided JSON data and provide insights such as summary statistics, trends, or key observations.
        
        The data is:
        {data}
        
        Answer:
        """
        PROMPT_ANALYSIS = PromptTemplate(ANALYSIS_TEXT)
        
        response = llm.complete(PROMPT_ANALYSIS.format(data=json.dumps(parsed_data)))
        return f"Analysis: {response.text}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:
        return f"Error: {str(e)}"
    
analyze_data_tool = FunctionTool.from_defaults(
    fn=analyze_data,
    name="analyze_data",
    description="Analyzes JSON data to provide insights such as summary statistics or trends."
)