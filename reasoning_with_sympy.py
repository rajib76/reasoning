# Author: Rajib Deb
# Date: 30-Nov-2024
# In this example, I tried a different technique where I told the model
# to write a sympy program to solve the problem. However, this program failed.
# The sympy program was right, but it did not consider all options
import os

from dotenv import load_dotenv
from openai import Client

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = Client()

messages = []


def answer_a_puzzle(model, system_prompt, human_prompt):
    if model == "o1-preview":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": human_prompt}]
        )
        final_response = response.choices[0].message.content
        return final_response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": human_prompt}]
        )
        final_response = response.choices[0].message.content
        return final_response


if __name__ == "__main__":
    model="gpt-4o-2024-08-06"
    system_prompt_template = """
    You are an excellent reasoner. You will be provided a context and a question that will require reasoning.
    Create a sympy based python program to solve the question. Sympy is a Python library for symbolic mathematics.
    Please think step by step to consider all options while creating the sympy program. Output only the program.
    DO NOT ADD ANYTHING ELSE.
    """
    human_prompt_template = """
    {context}
    {question}
    answer:
    """
    context = """
    I will be going to searworld in san diego.
    I need three tickets one for me, my wife and my daughter.
    The cost of anyday ticket is 119 USD per ticket and cost of a ticket for a particular ticket is 79 USD.
    With anyday ticket, there is an offer of buy one get one free.
    I want to visit Seaworld on a specific date.
    """
    question = "How should I buy the tickets to save the maximum amount of money"
    human_prompt = human_prompt_template.format(question = question,context=context)

    system_prompt = system_prompt_template


    response = answer_a_puzzle(model=model, system_prompt=system_prompt, human_prompt=human_prompt)

    print(response)
