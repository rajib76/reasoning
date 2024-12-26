import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


class Grading(BaseModel):
    grade: int = Field(description="rating for the trajectory")
    explanation: str = Field(description="explanation of the rating")


class Option(BaseModel):
    thought: str = Field(description="Thought related to the plan trajectory")
    option: str = Field(description="option based on the thought, which can be used to solve the input")


class ReasoningOutput(BaseModel):
    input: str = Field(description="Input from the user")
    options: List[Option]


class Reasoner(BaseModel):
    reasoner_system_prompt: str = """
    You are an expert in reasoning and planning. Given an input and list of previously 
    generated options, you generate at least four new options which are independent 
    of each other to address the solution of the input. 
    
    Please follow the below instruction for creating the options.
    ## Instructions:
    1. Carefully review the input and the previously generated options.
    2. Identify any errors or mistakes in the previously generated options.
    3. If there are any mistakes, modify the option to correct them in the new proposed option.
    4. Each new proposed option must build on top of previous options but must be independent and should be implementable stand-alone.
    5. Each option should have a description.
    6. Share the output in a JSON format
    """

    reasoner_user_prompt:str = """
    Here is the input and the previous options:
    
    ## Input:
    {input}
    ## Previous Options:
    {prev_steps}
    
    """

    grader_system_prompt_template: str = """
    You are an excellent grader.You will be grading a step generated to address a given input.
    You will rate the step on a scale of 1 to 10, where 1 is the worst and 10 is the best.
    A great step to address the given input:
    - must help advance the solution to address the given input
    - must be 100% accurate
    - must not have any irrelevant content
  
    A bad response does not meet one or more of the above requirements.
    """

    grader_prompt: str = """
    Here is the input and the step for you to grade:
    ## Input:
    {input}
    ## Step:
    {step}
    """

    def create_reasoining_steps(self, input, prev_steps):
        if prev_steps == "":
            prev_steps = "First iteration, hence not previous steps are avaliable"
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.reasoner_system_prompt},
                {"role": "user", "content": self.reasoner_user_prompt.format(input=input, prev_steps=prev_steps)},
            ],
            response_format=ReasoningOutput,
        )

        return completion.choices[0].message.parsed

    def grade_options(self, input: str, step: str):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.grader_system_prompt_template},
                {"role": "user", "content": self.grader_prompt.format(step=step, input=input)},
            ],
            response_format=Grading,
        )

        return completion.choices[0].message.parsed

    def main(self, input, depth=3, top_k=4):
        i = 0  # Determines the depth of the thought tree
        prev_steps = ""  # initially no previous steps
        final_options = []  # These will have the final set of options
        while i < depth:
            all_options = []  # At each depth level, this will collect the options
            grades = []  # The gradings of the option will be in this list
            grade = {}  # the grade will have two keys rate and option
            output = self.create_reasoining_steps(input=input, prev_steps=prev_steps)
            options = output.options
            # Store the steps or the options in the list
            for option in options:
                all_options.append(option)

            # Now call the grader to grade each step/option
            for option in all_options:
                rating = self.grade_options(input=input, step=option)
                grade["rate"] = rating.grade
                grade["option"] = option
                grades.append(grade)
                grade = {}

            # sort the list so that the top graded steps/options are at the top
            grades = sorted(grades, key=lambda k: k['rate'], reverse=True)
            # Take the top_k steps with the highest grading
            for out_grade in grades[0:top_k]:
                # Add the options in the final_option list
                final_options.append(out_grade["option"].option)
                # prepare the previous steps for the next iteration of step creation
                prev_steps = prev_steps + "\n" + out_grade["option"].option + "\n"

            i = i + 1

        # these steps/options can now be given to the worker model to solve the problem
        # for option in final_options:
        #     print(option)
        #     print("----")

        return final_options[-top_k:]


if __name__ == "__main__":
    input = """Next door to me live four brothers of different heights. 
    Their average height is 74 inches, and the difference in 
    height among the first three men is two inches. 
    The difference between the third and the fourth man is six inches.
    Provide the solution options to identify the height of each brother."""
    re = Reasoner()
    final_options = re.main(input=input)
    # print(final_options)
    for option in final_options:
        print(option)
        print("----")
