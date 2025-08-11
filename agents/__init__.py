class Agent:
    def __init__(self, name: str, instructions: str, model: str | None = None):
        self.name = name
        self.instructions = instructions
        self.model = model
