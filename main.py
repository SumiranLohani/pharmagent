import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum

# Assuming these are in the project structure as provided
# To make this example runnable, I've created dummy versions of these files.
from agents.planning_agent import PlanningAgent
from utils.communication_bus import CommunicationBus

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Workflow API",
    description="An API to trigger an agentic workflow, converted from Flask to FastAPI.",
    version="1.0.0"
)

# --- Configuration ---
# Switchable deployment configuration using an environment variable
# Set DEPLOYMENT_ENV=local for local Uvicorn server
# Unset or set to anything else for AWS Lambda deployment
IS_LOCAL_DEPLOYMENT = os.environ.get("DEPLOYMENT_ENV", "local") == "local"


# --- Pydantic Models ---
# Use Pydantic to define the structure of your request body.
# This provides automatic validation and documentation.
class ExecutionRequest(BaseModel):
    """Defines the expected JSON body for the /execute endpoint."""
    request: str


# --- API Endpoint ---
@app.post("/execute")
async def execute_workflow(execution_request: ExecutionRequest):
    """
    API endpoint to trigger the agentic workflow.

    FastAPI uses the type hint `execution_request: ExecutionRequest` to:
    1. Expect a JSON body that matches the ExecutionRequest model.
    2. Automatically validate the request. If 'request' is missing or not a string,
       it will return a 422 Unprocessable Entity error with details.
    3. Make the parsed and validated data available in the `execution_request` variable.
    """
    user_request = execution_request.request

    try:
        # Initialize the communication bus
        bus = CommunicationBus()
        bus.set('user_request', user_request)

        # Start the workflow with the Planning Agent
        # The agent's `run` method is synchronous, so we call it directly.
        # If it were an async method, we would `await` it.
        planning_agent = PlanningAgent(bus)
        final_report = planning_agent.run()

        return final_report
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# --- Deployment Logic ---
if IS_LOCAL_DEPLOYMENT:
    # This block runs the application using Uvicorn for local development.
    # To run: `python main.py` in your terminal.
    if __name__ == "__main__":
        print("Running in local development mode.")
        uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # This block configures the app for AWS Lambda using Mangum.
    # Mangum is an adapter for running ASGI applications in a serverless environment.
    print("Configured for AWS Lambda deployment.")
    handler = Mangum(app)

    # The `lambda_handler` function is the entry point for AWS Lambda.
    # In your Lambda configuration, you would point to `main.lambda_handler`.
    def lambda_handler(event, context):
        return handler(event, context)