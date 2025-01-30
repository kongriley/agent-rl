
from dataclasses import dataclass
from agentdojo.functions_runtime import FunctionsRuntime, TaskEnvironment

@dataclass
class Suite:
    environment: TaskEnvironment
    runtime: FunctionsRuntime
    system_description: str
    name: str

