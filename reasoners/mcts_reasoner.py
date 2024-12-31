import math
import os
import random

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from reasoners.base import PlanningAgent

EPSILON = 1e-6

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


class Thought(BaseModel):
    thought: str = Field(description="Follow-up thought based on input and previous thoughts")


class Grading(BaseModel):
    grade: int = Field(description="rating for the thought")
    explanation: str = Field(description="explanation of the rating")


grader_system_prompt_template: str = """
    You are an excellent grader.You will be grading a thought based on the input and 
    the previous thoughts, if provided,  to address the input.

    You will rate the thought on a scale of 1 to 10, where 1 is the worst and 10 is the best.
    A great thought to address the solution of the given input:
    - must help advance the solution to address the given input
    - must be 100% accurate
    - must not have any irrelevant content

    A poor thought does not meet one or more of the above requirements.
    """

thought_creation_prompt = """
Role: System 2 thinker
Backstory: You are a system 2 thinker.Given an input and a list of previous thought, you generate a follow up 
thought that can advance the solution to solve the given input
Task: Generate a follow-up thought based on the input and the list of previous thoughts
Instructions:
- Always think step by step
- Any follow-up thought must build on top of previous thoughts
- If no more follow-up thoughts are required, ONLY reply with 'END'. DO NOT ADD ANYTHING ELSE.

You will always share your response in a JSON format based on the format provided to you/
"""


class ThoughtNode:
    """
    This will be the nodes on the monte carlo tree
    """

    def __init__(self, input="", parent=None):
        self.input = input  # We will store the input here
        self.parent = parent
        self.children = []

        self.value = 0.0  # Accumulated reward (sum of ratings)
        self.visits = 0  # Number of times this node was visited
        self.thought = ""  # Thought generated will be stored here
        self.depth = self.parent.depth + 1 if parent else 0  # Till what depth we want to explore
        # If parent exists, we optionally append the node as a child
        if parent is not None:
            parent.children.append(self)

    def backpropagate(self, reward):
        """
        Propagate the reward up the tree, updating `value` and `visits`
        for each ancestor node (including self).
        """
        node = self
        while node is not None:
            node.visits += 1  # increment the visit to the node
            # In actual MCTS, the value is the win ratio
            # here it is the average of rating of the thought
            # This way of rating the thought may not be correct.
            # Need to think of a better way.
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def traverse_up_to_root(self):
        """
        Traverse from the given node up through its parents,
        collecting nodes on the way, until we reach a node
        whose parent is None (the root).
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path


class MCTSReasoner(PlanningAgent):

    def __init__(self, no_of_simulations=2, exploration_constant=1.41, verbose=False):
        """
        :param no_of_simulations: Number of MCTS simulations
        :param exploration_constant: UCT exploration constant
        :param verbose: Whether to print debug info
        """
        super().__init__()
        self._no_of_simulations = no_of_simulations
        self._exploration_constant = exploration_constant
        self._verbose = verbose # Need to implement this

        self._root = None
        self._max_depth = 4 # Default is 4

    def _is_no_more_thought(self, node):
        return node.depth >= self._max_depth or "END" in node.thought

    def _return_best_node(self, input,expand=2,visualize_tree=True):
        # Start from the root node
        root = ThoughtNode(input=input, parent=None)
        self._root = root
        thought_nodes = []

        for _ in range(self._no_of_simulations):
            node = root

            while not self._is_no_more_thought(node) and len(node.children) > 0:
                # This is the place where the Upper Control Boundary is being
                # calculated
                choices_weights = [
                    (child.value / (child.visits + EPSILON))
                    + self._exploration_constant
                    * math.sqrt(
                        2.0 * math.log(node.visits + EPSILON) / (child.visits + EPSILON)
                    )
                    for child in node.children
                ]
                node = node.children[choices_weights.index(max(choices_weights))]

            while not self._is_no_more_thought(node):
                if len(node.children) == 0:
                    self._expand(node, expand)
                node = random.choice(node.children)
                rating = self._rate_thought(node)
                node.backpropagate(rating.grade)
            thought_nodes.append(node)
        best_thought_node = max(thought_nodes, key=lambda node: node.value)

        if visualize_tree:
            self.print_tree_ascii(root)

        return best_thought_node

    def print_tree_ascii(self,node, prefix="", is_last=True):
        """
        Recursively print the tree in ASCII form.

        :param node:    The current ThoughtNode.
        :param prefix:  The string prefix for the current level (e.g., '│   ').
        :param is_last: Whether the current node is the last child of its parent.
        """
        # Determine the branch to use for ASCII lines
        branch = "└── " if is_last else "├── "
        # Print the current node
        print(prefix + branch + f"{node.thought} (depth={node.depth}, value={node.value}, visits={node.visits})")

        # Update prefix for children
        # If it's the last child, the prefix gets "    ", otherwise it gets "│   "
        new_prefix = prefix + ("    " if is_last else "│   ")

        # Iterate over children
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            # For each child, determine if it's the last in the list
            is_child_last = (i == (child_count - 1))
            self.print_tree_ascii(child, prefix=new_prefix, is_last=is_child_last)

    def _rate_thought(self, node):
        if node.parent is None:
            parent_thought = ""
        else:
            parent_thought = node.parent.thought
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": grader_system_prompt_template},
                {"role": "user", "content": "input:\n" +
                                            node.input + "\n" + "previous thoughts\n" +
                                            parent_thought + "new thought: \n" + node.thought},
            ],
            response_format=Grading,
        )

        return completion.choices[0].message.parsed

    def _expand(self, node, expand):
        """
        Expands a node by generating its children. In a typical MCTS for text generation,
        this might mean generating possible 'next steps' or partial answers.
        Here we'll just generate some placeholder children.
        """
        # Example: randomly create 1-3 new child nodes to simulate possible next states
        childs = []
        for i in range(expand):
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": thought_creation_prompt},
                    {"role": "user",
                     "content": "input:\n" + node.input + "\n" + "previous thoughts\n" + node.thought},
                ],
                response_format=Thought,
            )

            response = completion.choices[0].message.parsed
            child = ThoughtNode(input=node.input, parent=node)
            child.thought = response.thought
            rating = self._rate_thought(child)
            child.backpropagate(rating.grade)
            childs.append(child)

        return childs

    def generate_reasoning(self,input,expand=2,visualize_tree=True):
        best_thought_node = self._return_best_node(input=input,expand=expand)
        return best_thought_node
