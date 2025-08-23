from agents import Agent
from pydantic import BaseModel
from tools import run_training_script

class TrainingResultOutput(BaseModel):
    success: bool
    error: str

trainer = Agent(
    name="Training Runner",
    instructions="""You are an expert in running neural network training experiments.
    Your task is to:
    1. Use run_training_script(name, script_path) where script_path is ONLY the file path
    2. Example: run_training_script(name="delta_net_dual_path", script_path="scripts/train.sh")
    3. DO NOT include "bash" in script_path - the function handles bash execution
    4. If training succeeds, set success=True and leave error empty
    5. If training fails, set success=False and analyze the root cause
       
    The function signature is: run_training_script(name: str, script_path: str)
    Pass script_path="scripts/train.sh", NOT "bash scripts/train.sh".""",
    tools=[run_training_script],
    output_type=TrainingResultOutput,
    model="gpt-5"
)
