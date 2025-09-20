from agents.execution_agent import ExecutionAgent
from utils.communication_bus import CommunicationBus

class PlanningAgent:
    """
    The Planning Agent receives the user's request and creates a plan of execution.
    """
    def __init__(self, bus: CommunicationBus):
        self.bus = bus

    def run(self):
        user_request = self.bus.get('user_request')
        
        # In a real-world scenario, a language model would be used to generate this plan
        # based on the user's request.
        plan = self._generate_plan(user_request)
        
        self.bus.set('plan', plan)
        self.bus.set('next_agent', 'ExecutionAgent')

        # Start the execution workflow
        execution_agent = ExecutionAgent(self.bus)
        return execution_agent.run()

    def _generate_plan(self, user_request):
        # This is a simplified plan generation. A more advanced implementation
        # would use an LLM to parse the user_request and create a dynamic plan.
        plan = []
        if "2D QSAR" in user_request:
            plan.append({"agent": "ExecutionAgent", "tool": "qsar_2d", "params": {}})
        if "3D QSAR" in user_request:
            plan.append({"agent": "ExecutionAgent", "tool": "qsar_3d", "params": {}})
        if "molecular docking" in user_request:
            plan.append({"agent": "ExecutionAgent", "tool": "molecular_docking", "params": {}})
        
        return plan