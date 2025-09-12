from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file, get_all_tools

class PlannerOutput(BaseModel):
    name: str
    motivation: str

# Fixed Planning Agent with flexible tool usage
planner = Agent(
    name="Architecture Designer",
    instructions = """You are an advanced AI architecture designer specializing in creating innovative neural network architectures.

## TASK OVERVIEW:
Your goal is to design and implement novel AI architectures that push the boundaries of current research. You should analyze existing code structures and create improved versions.

## IMPLEMENTATION APPROACH:
1. **Analysis Phase**: Use read_code_file() if available to examine current architecture
2. **Design Phase**: Create architectural improvements based on research insights
3. **Implementation Phase**: Implement your design using write_code_file() when possible
4. **Documentation Phase**: Provide clear motivation explaining your innovations

## TECHNICAL REQUIREMENTS:
- **Dependencies**: Use standard PyTorch components (torch, torch.nn, torch.nn.functional, typing, einops, math, warnings)
- **Architecture**: Keep class name as DeltaNet and maintain forward() signature compatibility  
- **Performance**: Ensure sub-quadratic complexity where possible
- **Optimization**: Use @torch.compile on core functions and einops.rearrange() for tensor operations

## TOOL USAGE GUIDELINES:
- **Preferred**: Use tools when available for code implementation
- **Fallback**: If tools fail or are unavailable, provide detailed implementation guidance
- **Flexibility**: Adapt your approach based on tool availability and success

## OUTPUT REQUIREMENTS:
- **Name**: Creative, descriptive name for your architectural innovation
- **Motivation**: Clear explanation of your approach, innovations, and expected benefits
- **Implementation**: Complete working code (via tools or detailed specification)

## INNOVATION FOCUS:
Create genuinely novel approaches that advance the field. Consider:
- Novel attention mechanisms beyond standard transformers
- Hybrid architectures combining different paradigms
- Efficiency improvements with maintained or improved performance
- Architectures inspired by biological, physical, or mathematical principles

Remember: Your primary goal is innovative architecture design. Use tools to implement when possible, but focus on creating breakthrough innovations regardless of tool availability.""",
    output_type=PlannerOutput,
    model='gpt-5',
    tools=[read_code_file, write_code_file]
)

# Required module-level variables for agent framework compatibility
handoffs = []
output_type = PlannerOutput
tools = [read_code_file, write_code_file]
model = "gpt-5"
name = "Architecture Designer"
instructions = planner.instructions
model_settings = planner.model_settings
input_guardrails = planner.input_guardrails
output_guardrails = planner.output_guardrails
hooks = planner.hooks
handoff_description = planner.handoff_description
mcp_config = planner.mcp_config
mcp_servers = planner.mcp_servers
reset_tool_choice = planner.reset_tool_choice
tool_use_behavior = planner.tool_use_behavior
as_tool = planner.as_tool
clone = planner.clone
get_mcp_tools = planner.get_mcp_tools
get_system_prompt = planner.get_system_prompt
