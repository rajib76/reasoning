from reasoners.mcts_reasoner import MCTSReasoner

if __name__ == "__main__":
    agent = MCTSReasoner(no_of_simulations=4, exploration_constant=1.41, verbose=True)
    # input = "How old was Rajeev Gandhi, when Indira Gandhi became Indian prime minister"
    input = "What is 2+2"
    best_node = agent.generate_reasoning(input)

    paths = best_node.traverse_up_to_root()
    print("path ", paths)
    for path in paths:
        print(path.thought)
        print(path.parent)
        print("------------------")
