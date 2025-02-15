import os
import openai
import asyncio
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with latest SDK
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

TARGET_AGENT_MODEL = "gpt-4o"
APPROXIMATE_AGENT_MODEL = "gpt-4o-mini"



# Define Pydantic model for structured response
class Step(BaseModel):
    explanation: str
    output: str


class PlanningSteps(BaseModel):
    steps: list[Step]


# Prompt Templates
target_agent_prompt = """
Given the task: 
task: /n
{task} /n 
verify and only improve upon the last action step, if it is significantly incorrect, otherwise output the step AS IS:
history:
{history} /n

* REMEMBER * only output the step. DO NOT ADD ANYTHING ELSE.
"""

approx_agent_prompt = """
Given the task: 
task: /n
{task} /n
generate only one next immediate action step based on previous steps.The step must be atomic and actionable.
Each subsequent step must be incremental over the previous steps with no overlap with previous steps.
If you are in the last step, just output "NO STEP"
history: /n
{history} /n

* REMEMBER * to generate the next action step with ZERO overlap with previous steps.
"""


async def call_openai(prompt, response_model, MODEL,temperature=0.7):
    """ Calls OpenAI GPT-4o and returns structured JSON output using Pydantic validation. """
    response = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a structured output assistant. Respond in JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format=response_model,  # Use structured output
        temperature=temperature
    )

    try:
        return response.choices[0].message.parsed  # Extract structured response
    except ValidationError as e:
        print("Pydantic Validation Error:", e)
        return None  # Handle errors gracefully


async def approximation_agent(task, history):
    """ The approximation agent (A) generates fast speculative steps. """
    prompt = approx_agent_prompt.format(task=task, history=history)
    print("\n=== Calling Approximation Agent ===")
    print(f"Task: {task}")
    print(f"History:\n{history}")
    print("==============================\n")
    MODEL = APPROXIMATE_AGENT_MODEL
    return await call_openai(prompt, PlanningSteps, MODEL,temperature=0.5)


async def target_agent(task, history):
    """ The target agent (T) verifies and corrects speculative steps. """
    prompt = target_agent_prompt.format(task=task, history=history)

    print("\n=== Calling Target Agent ===")
    print(f"Task: {task}")
    print(f"History:\n{history}")
    print("==============================\n")
    MODEL = TARGET_AGENT_MODEL
    return await call_openai(prompt, PlanningSteps, MODEL,temperature=0.3)


async def speculative_planning(task, max_steps=5, k=2):
    """ Implements the speculative planning algorithm with approximation and target agents. """
    steps = []  # Stores completed steps

    for i in range(max_steps):
        # Ensure history is updated **before** calling the approximation agent
        history = "\n".join([step.output for step in steps]) if steps else "No previous steps."

        # Generate the next speculative step
        approx_response = await approximation_agent(task, history)

        if approx_response is None or not approx_response.steps:
            print(f"Error: No valid output from approximation agent at step {i}. Skipping.")
            break

        approx_step = approx_response.steps[0]  # Take the first step
        print(f"Step {i}: Approximation Agent Generated -> {approx_step.output}")

        # Append speculative step to steps list
        steps.append(approx_step)

        # Update history for target agent
        history = "\n".join([step.output for step in steps])

        # Start verifying after `k` steps
        if i >= k:
            verified_response = await target_agent(task, history)

            if verified_response is None or not verified_response.steps:
                print(f"Error: No valid output from target agent at step {i}. Using speculative step.")
                continue

            verified_step = verified_response.steps[0]  # Take first verified step

            if verified_step.output.strip().lower() != approx_step.output.strip().lower():
                print(f"Step {i}: Mismatch detected! Using verified step instead.")

                # Replace incorrect speculative step
                steps[-1] = verified_step
            else:
                print(f"Step {i}: No Mismatch detected! Using same step created by approximation agent.")

            print(f"Step {i}: Final Step -> {steps[-1].output}")

        if approx_step.output == "NO STEP":
            print("\nNo more steps to generate. Stopping early.")
            break

    print("\nFinal Steps:")
    for idx, step in enumerate(steps):
        print(f"Step {idx}: {step.output}")

    return steps


# Example Usage
task_description = "I want to buy a house in San Jose. Please help me plan my buying process. I have a budget of $1000,000."
asyncio.run(speculative_planning(task_description, max_steps=5, k=2))
