from abc import abstractmethod
class PlanningAgent():

    def __init__(self):
        self.__module__ = "Planner"

    @abstractmethod
    def generate_reasoning(self,input,**kwargs):
        pass
