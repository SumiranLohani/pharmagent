
# PharmAgent-Microservices

```
PharmAgent-Microservices/
│
├── orchestrator/
│   ├── main_api_orchestrator.py      # The main agent you run.
│   └── README.md                     # Instructions on how to run the orchestrator.
│
├── services/
│
│   ├── qsar_2d_service/
│   │   ├── qsar_2d_api.py            # The FastAPI wrapper.
│   │   ├── agent_2d_qsar_ai.py       # Your original, core logic script.
│   │   ├── requirements.txt          # Dependencies specific to this service.
│   │   └── Dockerfile                # (Optional but recommended) For containerizing the service.
│
│   ├── qsar_3d_service/
│   │   ├── qsar_3d_api.py            # The FastAPI wrapper for the 3D tool.
│   │   ├── requirements.txt
│   │   └── Dockerfile
│
│   └── docking_service/
│       ├── docking_api.py            # The FastAPI wrapper for the docking tool.
│       ├── requirements.txt
│       └── Dockerfile
│
├── docker-compose.yml                # (Optional but recommended) A single file to start all services at once.
└── requirements.txt                  # Top-level dependencies, mainly for the orchestrator.
```