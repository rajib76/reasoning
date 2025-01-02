import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from reasoners.base import PlanningAgent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

thought_creation_prompt = """
Role: System 2 thinker
Backstory: You are a system 2 thinker. Given an input and a list of previous plans, you generate a follow-up 
plan that can advance the solution to solve the given input.
Task: Generate a follow-up plan based on the input and the list of previous plans.
Instructions:
- Always think step by step
- Any follow-up plan must build on top of previous plans
- If no more follow-up plans are required, ONLY reply with 'TERMINATE'. DO NOT ADD ANYTHING ELSE.

You will always share your response in a JSON format based on the format provided to you
"""


class Citation(BaseModel):
    citation: str = Field(descripton="name of the section from where answer is derived")


class Responses(BaseModel):
    answer: str = Field(description="answer to the question based on the context")
    citations: List[Citation] = Field(description="list of the section used to answer the question")


class Plan(BaseModel):
    plan: str = Field(description="Plan to solve a given task")
    plan_summary: str = Field(description="One word that summarizes the plan")


class Plans(BaseModel):
    plans: List[Plan] = Field(description="List of alternate plans to solve and address a provided input")


class PlanNode:
    def __init__(self, task: str, plan: str, plan_summary: str):
        self.task = task
        self.plan = plan
        self.plan_summary = plan_summary
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class PlanTree(PlanningAgent):

    def __init__(self, task: str, plan: str, plan_summary: str, max_depth: int):
        super().__init__()
        self.root = PlanNode(task, plan, plan_summary)
        self.max_depth = max_depth

    def build_tree(self, node: PlanNode, current_depth: int):
        if current_depth >= self.max_depth:
            return

        plans = self.generate_plan(node.task, node.plan, depth=1)  # Generate {depth} thoughts for each node
        for plan in plans:
            child_node = PlanNode(task=node.task, plan=plan["plan"], plan_summary=plan["plan_summary"])
            node.add_child(child_node)
            self.build_tree(child_node, current_depth + 1)

    def generate_plan(self, prompt: str, previous_plan: str, depth: int) -> List[{}]:
        """
        Generates a list of thoughts based on the given prompt and the previous plan.
        """
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": thought_creation_prompt},
                {"role": "user",
                 "content": f"Generate {depth} plans based on: {prompt}\nPrevious plan: {previous_plan}"},
            ],
            response_format=Plans,
        )

        plans = completion.choices[0].message.parsed.plans
        return [{"plan": plan.plan, "plan_summary": plan.plan_summary} for plan in plans]

    def print_ascii_tree(self, node: PlanNode, prefix: str = "", is_last: bool = True):
        """
        Prints an ASCII representation of the tree starting from the given node.
        """
        connector = "└── " if is_last else "├── "
        plan_step = f"{prefix}{connector}{node.plan_summary}: {node.plan}"
        prefix += "    " if is_last else "│   "
        with open("plan.txt", "a") as f:
            f.write(plan_step)
        print(plan_step)
        # time.sleep(1)

        child_count = len(node.children)
        for i, child in enumerate(node.children):
            self.print_ascii_tree(child, prefix, i == child_count - 1)

    def generate_reasoning(self, input, **kwargs):
        pass

    def generate_response(self, question=""):
        with open("plan.txt") as f:
            plan = f.readlines()

        prompt = """
        Follow the instructions to provide an answer to the question based on the context provided.
        
        Instructions:
        {instruction}
        
        Question:
        {question}
        
        Context:
        Section 1:
        "Recent studies on renewable energy highlight the increasing efficiency of solar panels, 
        which now exceed 22% efficiency in commercial models. This is achieved through advancements in 
        photovoltaic cell design and material science. Researchers are focusing on reducing production 
        costs while improving durability and sustainability."
    
        Section 2:
        "Wind energy has become a vital component of global renewable energy efforts. 
        Offshore wind farms are particularly promising, offering higher wind speeds and 
        consistent energy generation. However, challenges such as high installation costs and 
        maintenance in harsh ocean environments remain significant barriers."
    
        Section 3:
        "Battery storage technology is critical to addressing the intermittency of 
        renewable energy sources. Lithium-ion batteries dominate the market, but research into 
        solid-state batteries shows potential for higher energy density, faster charging times, and improved safety. 
        These innovations could revolutionize grid storage and electric vehicles."
        
        """
        print(plan)
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": thought_creation_prompt},
                {"role": "user",
                 "content": prompt.format(question=question,instruction=plan)},
            ],
            response_format=Responses,
        )

        responses = completion.choices[0].message.parsed

        print("responses ", responses)

        # all_response = [{"response": response.answer, "citation": [citation for citation in response.citations]} for response in responses]
        return responses


if __name__ == "__main__":
    prompt = """
    The task is to generate an answer for a question based on the below context. 
    The context has multiple sections. The answer needs to be formulated based on all relevant sections, 
    but we should not mix the content from different sections. Each section's content must remain distinct, 
    and the answer should clearly indicate which section is being referenced. Citations must be included, 
    specifying the section and relevant details to support the answer.

    Context:
    Section 1:
    "Recent studies on renewable energy highlight the increasing efficiency of solar panels, 
    which now exceed 22% efficiency in commercial models. This is achieved through advancements in 
    photovoltaic cell design and material science. Researchers are focusing on reducing production 
    costs while improving durability and sustainability."

    Section 2:
    "Wind energy has become a vital component of global renewable energy efforts. 
    Offshore wind farms are particularly promising, offering higher wind speeds and 
    consistent energy generation. However, challenges such as high installation costs and 
    maintenance in harsh ocean environments remain significant barriers."

    Section 3:
    "Battery storage technology is critical to addressing the intermittency of 
    renewable energy sources. Lithium-ion batteries dominate the market, but research into 
    solid-state batteries shows potential for higher energy density, faster charging times, and improved safety. 
    These innovations could revolutionize grid storage and electric vehicles."

    Question:
    How do advancements in renewable energy technologies address the challenges of efficiency, 
    consistency, and storage?
    """

    plan_tree = PlanTree(task=prompt, plan="Initial Plan", plan_summary="Overview", max_depth=1)
    plan_tree.build_tree(plan_tree.root, 0)
    plan_tree.print_ascii_tree(plan_tree.root)

    question = """
    How do advancements in renewable energy technologies address the challenges of efficiency, 
    consistency, and storage?
    """
    response = plan_tree.generate_response(question=question)
    print(response.answer)
    print(response.citations)




