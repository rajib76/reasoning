import random


class NaturalProgram:
    """Represents a goal decomposition with linguistic hints and a concrete plan."""

    def __init__(self, goal, hint, decomposition):
        self.goal = goal  # Main goal (e.g., "Make taco")
        self.hint = hint  # User-provided hint (e.g., "Combine salsa and meat")
        self.decomposition = decomposition  # List of sub-goals or primitive actions

    def __repr__(self):
        return f"NaturalProgram(goal='{self.goal}', hint='{self.hint}', decomposition={self.decomposition})"


class PlannerAgent:
    """Hierarchical planner agent that learns from user hints and adapts over generations."""

    def __init__(self):
        self.library = {}  # Task decomposition library

    def add_program(self, goal, hint, decomposition):
        """Adds a new task decomposition to the library."""
        if goal not in self.library:
            self.library[goal] = []
        self.library[goal].append(NaturalProgram(goal, hint, decomposition))

    def get_possible_decompositions(self, goal):
        """Returns possible decompositions for a goal."""
        return self.library.get(goal, [])

    def execute_plan(self, goal, context):
        """
        Recursively executes a plan for a goal using hierarchical decomposition.
        If a decomposition exists, it follows the sub-goals. If not, it asks the user.
        """
        print(f"Executing: {goal} in context {context}")

        # Check if the goal has been learned
        if goal in self.library:
            possible_decompositions = self.get_possible_decompositions(goal)

            # Choose a decomposition dynamically based on the context
            selected_decomposition = random.choice(possible_decompositions).decomposition
            print(f"Using decomposition: {selected_decomposition}")

            for step in selected_decomposition:
                if step in self.library:
                    self.execute_plan(step, context)  # Recursive execution
                else:
                    print(f"Performing primitive action: {step}")  # Execute atomic action
            return True

        else:
            # No known decomposition - ask for a hint
            print(f"Unknown goal: {goal}. Please provide a hint on how to accomplish it.")
            hint = input(f"How can '{goal}' be achieved? ")
            sub_goals = input("Enter sub-goals (comma-separated): ").split(", ")

            # Store the new decomposition
            self.add_program(goal, hint, sub_goals)
            return self.execute_plan(goal, context)  # Retry with new knowledge


# Example Usage: Teaching and Executing Plans

# Initialize the planner agent
agent = PlannerAgent()

# Teaching the agent how to make a taco
agent.add_program("Make taco", "Combine salsa and meat", ["Make salsa", "Add meat"])

# Teaching how to make salsa in two different ways
agent.add_program("Make salsa", "Use tomato and red pepper", ["Chop tomato", "Chop red pepper", "Mix"])
agent.add_program("Make salsa", "Use avocado and green pepper", ["Chop avocado", "Chop green pepper", "Mix"])

# Execute a goal in a new context (this simulates learning over generations)
context1 = {"ingredients": ["tomato", "red pepper", "meat"]}
context2 = {"ingredients": ["avocado", "green pepper", "meat"]}

# First execution with known ingredients
print("\n--- Running in Context 1 ---")
agent.execute_plan("Make taco", context1)

# Second execution with different ingredients (adapting from previous experience)
print("\n--- Running in Context 2 ---")
agent.execute_plan("Make taco", context2)

# If an unknown goal is encountered, the user will be prompted to provide a hint
print("\n--- Running an Unknown Goal ---")
agent.execute_plan("Make burrito", {"ingredients": ["beans", "tortilla", "meat"]})
