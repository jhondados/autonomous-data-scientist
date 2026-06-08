"""Autonomous data scientist agent."""
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import io

@tool
def load_and_profile_data(filepath: str) -> str:
    """Load CSV and return statistical profile."""
    df = pd.read_csv(filepath)
    profile = f"Shape: {df.shape}\nColumns: {list(df.columns)}\nDtypes:\n{df.dtypes.to_string()}\nDescribe:\n{df.describe().to_string()}\nMissing:\n{df.isnull().sum().to_string()}"
    return profile

@tool
def run_python_analysis(code: str, data_path: str) -> str:
    """Execute Python analysis code on dataset and return results."""
    import subprocess, tempfile
    script = f"import pandas as pd\ndf = pd.read_csv('{data_path}')\n{code}"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script); fname = f.name
    result = subprocess.run(['python', fname], capture_output=True, text=True, timeout=60)
    return result.stdout[:2000] + (result.stderr[:500] if result.stderr else "")

def create_autonomous_scientist():
    llm = ChatVertexAI(model_name="gemini-1.5-pro-002")
    tools = [load_and_profile_data, run_python_analysis]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data scientist. Analyze data thoroughly. Always profile first, then analyze, then interpret."),
        ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=20)
