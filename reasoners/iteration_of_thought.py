# Based on the paper
# https://arxiv.org/pdf/2409.12618
import os

import time

import openai
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

class InnerDialogueAgent:
    """
    Inner Dialogue Agent (IDA) generates refined prompts based on previous responses.
    """

    def __init__(self):
        pass

    def generate_prompt(self, query, previous_response):
        """
        Generate a refined prompt based on the user's query and the last response.
        """
        return f"Validate and refine the following response, if needed, considering the context: {previous_response}\nUser query: {query}. If the answer is complete, explicitly state 'Final Answer' or 'Sufficient Answer'."


class LLMAgent:
    """
    LLM Agent (LLMA) processes prompts and generates responses.
    """

    def __init__(self, model="gpt-4o"):
        self.model = model

    def generate_response(self, prompt):
        """
        Generate a response from GPT-4o based on the refined prompt.
        """
        print("--prompt--")
        print(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system",
                       "content": "You are an advanced AI assistant helping with iterative thought reasoning."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content


class IterationOfThought:
    """
    Implements the IoT framework using IDA and LLMA.
    """

    def __init__(self, max_iterations=5, mode="AIoT"):
        self.ida = InnerDialogueAgent()
        self.llm = LLMAgent()
        self.max_iterations = max_iterations
        self.mode = mode  # "AIoT" or "GIoT"

    def iterate_response(self, query):
        """
        Run the iterative reasoning loop based on AIoT or GIoT.
        """
        response = self.llm.generate_response(query)  # Initial response

        for i in range(self.max_iterations):
            print(f"Iteration {i + 1}: {response}\n")

            if self.mode == "AIoT":
                # AIoT: Stop if response seems sufficient
                if "final answer" in response.lower() or "sufficient answer" in response.lower():
                    break

            refined_prompt = self.ida.generate_prompt(query, response)
            response = self.llm.generate_response(refined_prompt)
            time.sleep(1)  # Avoid excessive API calls

        return response


if __name__ == "__main__":
    iot = IterationOfThought(max_iterations=3, mode="AIoT")  # Change mode to "GIoT" for fixed iterations
    user_query = "How many r are there in Strawberry?"
    final_response = iot.iterate_response(user_query)

    print("\nFinal Refined Response:")
    print(final_response)
