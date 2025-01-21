"""
FastAPI Gateway for AlphaAgents.

Exposes the multi-agent framework as a RESTful API for external integration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from orchestration import AlphaGroupChat
from portfolio import PortfolioBuilder
from orchestration.monitor import GovernanceMonitor
import time

app = FastAPI(title="AlphaAgents API", version="1.0.0")
monitor = GovernanceMonitor()

class AnalysisRequest(BaseModel):
    tickers: List[str]
    risk_profile: str = "neutral"
    use_neural: bool = True

class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    results: Optional[Dict[str, Any]] = None

# In-memory store for background tasks (replace with Redis in prod)
jobs = {}

@app.get("/")
async def health_check():
    return {"status": "online", "engine": "AlphaAgents Core v1.0", "metrics": monitor.get_summary()}

@app.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {"status": "processing", "start_time": time.time()}
    
    background_tasks.add_task(run_agent_orchestration, job_id, request)
    
    return AnalysisResponse(request_id=job_id, status="processing")

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def run_agent_orchestration(job_id: str, request: AnalysisRequest):
    start_time = time.time()
    try:
        # Initialize AlphaGroupChat
        chat = AlphaGroupChat()
        portfolio_results = chat.run_sequential_analysis(request.tickers)
        
        # Build final portfolio
        builder = PortfolioBuilder(request.risk_profile)
        final_recs = builder.build_portfolio(portfolio_results)
        
        latency = time.time() - start_time
        monitor.log_interaction("FastAPI_Gateway", f"Analysis: {','.join(request.tickers)}", {
            "tokens": 4500, # Mock token usage
            "latency": latency,
            "status": "success"
        })
        
        jobs[job_id] = {
            "status": "completed",
            "tickers": request.tickers,
            "recommendations": final_recs,
            "latency_sec": latency,
            "timestamp": time.time()
        }
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
