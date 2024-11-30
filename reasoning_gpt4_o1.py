# Author: Rajib Deb
# Date: 30-Nov-2024
# In this code, I compare the reasoning capabilities of GPT-4 and O1-Series
# The problem statement provided to the models is:
# I will be going to searworld in san diego.
# I need three tickets one for me, my wife and my daughter.
# The cost of anyday ticket is 119 USD per ticket and cost of a ticket for a particular ticket is 79 USD.
# With anyday ticket, there is an offer of buy one get one free.
# I want to visit Seaworld on a specific date.
import os

from dotenv import load_dotenv
from openai import Client

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = Client()

messages = []


def answer_a_puzzle(model, system_prompt, human_prompt):
    """
    o1-preview does not take the system role, it only uses user role
    :param model: I will pass the model from the main module below
    :param system_prompt: The system prompt to be used while using GPT-4
    :param human_prompt: The human prompt to be used
    :return: Returns the output of the model
    """
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
    # Use this to test with GPT-4

    # model="gpt-4o-2024-08-06"
    # system_prompt_template = """
    # You are an excellent reasoner. You will be provided a context and a question that will require reasoning.
    # Please think step by step and answer the provided question.
    # """
    # human_prompt_template = """
    # {context}
    # {question}
    # answer:
    # """
    # context = """
    # I will be going to searworld in san diego.
    # I need three tickets one for me, my wife and my daughter.
    # The cost of anyday ticket is 119 USD per ticket and cost of a ticket for a particular ticket is 79 USD.
    # With anyday ticket, there is an offer of buy one get one free.
    # I want to visit Seaworld on a specific date.
    # """
    # question = "How should I buy the tickets to save the maximum amount of money"
    # human_prompt = human_prompt_template.format(question = question,context=context)
    #
    # system_prompt = system_prompt_template

    # Use this to test with o1-preview

    model = "o1-preview"
    system_prompt_template = """
    You will answer based on the context and the question provided."""

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
    human_prompt = human_prompt_template.format(question=question, context=context)

    system_prompt = ""

    response = answer_a_puzzle(model=model, system_prompt=system_prompt, human_prompt=human_prompt)

    print(response)
