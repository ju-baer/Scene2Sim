"""
Web application for Scene2Sim.
"""
import os
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

from ..core.scene import Scene
from ..core.simulator import Simulator
from ..io.loaders import load_image, load_video, quick_load
from ..analysis.metrics import MetricsCalculator
from ..render.three_viz import ThreeJSExporter

# Initialize FastAPI app
app = FastAPI(title="Scene2Sim", description="Advanced Scene Analysis and Simulation")

# Setup templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Global storage for processing jobs
processing_jobs: Dict[str, Dict[str, Any]] = {}
completed_scenes: Dict[str, Scene] = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main application page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process image/video file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    # Initialize job status
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'message': 'File uploaded successfully',
        'filename': file.filename,
        'file_path': tmp_file_path,
        'scene_id': None,
        'error': None
    }
    
    # Start background processing
    background_tasks.add_task(process_file_async, job_id, tmp_file_path, file.filename)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get processing job status."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.get("/scene/{scene_id}")
async def get_scene(scene_id: str):
    """Get scene data."""
    if scene_id not in completed_scenes:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    scene = completed_scenes[scene_id]
    return scene.to_dict()

@app.get("/scene/{scene_id}/metrics")
async def get_scene_metrics(scene_id: str):
    """Get scene analysis metrics."""
    if scene_id not in completed_scenes:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    scene = completed_scenes[scene_id]
    calculator = MetricsCalculator()
    metrics = calculator.calculate_scene_metrics(scene)
    
    return metrics.__dict__

@app.post("/scene/{scene_id}/simulate")
async def simulate_scene(scene_id: str, background_tasks: BackgroundTasks, 
                        duration: float = 10.0, enable_physics: bool = True):
    """Run simulation on scene."""
    if scene_id not in completed_scenes:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Generate simulation job ID
    sim_job_id = f"{scene_id}_sim_{uuid.uuid4().hex[:8]}"
    
    processing_jobs[sim_job_id] = {
        'status': 'running',
        'progress': 0.0,
        'message': 'Starting simulation...',
        'scene_id': scene_id,
        'type': 'simulation'
    }
    
    # Start simulation
    background_tasks.add_task(run_simulation_async, sim_job_id, scene_id, duration, enable_physics)
    
    return {"job_id": sim_job_id, "status": "running"}

@app.get("/simulation/{job_id}/result")
async def get_simulation_result(job_id: str):
    """Get simulation results."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    job = processing_jobs[job_id]
    
    if job['status'] != 'completed':
        return {"status": job['status'], "progress": job.get('progress', 0)}
    
    return job.get('result', {})

@app.get("/scene/{scene_id}/export/threejs")
async def export_threejs(scene_id: str):
    """Export scene to Three.js format."""
    if scene_id not in completed_scenes:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    scene = completed_scenes[scene_id]
    exporter = ThreeJSExporter()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        scene_data = exporter._scene_to_threejs(scene)
        json.dump(scene_data, tmp_file, indent=2)
        tmp_path = tmp_file.name
    
    return FileResponse(
        tmp_path,
        media_type='application/json',
        filename=f"scene_{scene_id}_threejs.json"
    )

@app.get("/viewer/{scene_id}")
async def get_3d_viewer(scene_id: str):
    """Get 3D viewer page for scene."""
    if scene_id not in completed_scenes:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    scene = completed_scenes[scene_id]
    exporter = ThreeJSExporter()
    
    # Generate scene data
    scene_data = exporter._scene_to_threejs(scene)
    
    # Create inline HTML viewer
    html_content = exporter.generate_html_viewer("", None)
    
    # Inject scene data directly into HTML
    scene_json = json.dumps(scene_data, indent=2)
    html_content = html_content.replace(
        "loadScene('')",
        f"createSceneFromData({scene_json})"
    )
    
    return HTMLResponse(content=html_content)

@app.get("/scenes")
async def list_scenes():
    """List all available scenes."""
    scenes_info = []
    
    for scene_id, scene in completed_scenes.items():
        calculator = MetricsCalculator()
        metrics = calculator.calculate_scene_metrics(scene)
        
        scenes_info.append({
            'id': scene_id,
            'source_path': scene.source_path,
            'object_count': metrics.object_count,
            'analysis_complete': scene.analysis_complete,
            'timestamp': scene.analysis_metadata.get('timestamp')
        })
    
    return {"scenes": scenes_info}

async def process_file_async(job_id: str, file_path: str, filename: str):
    """Process uploaded file asynchronously."""
    try:
        # Update status
        processing_jobs[job_id].update({
            'status': 'processing',
            'progress': 0.1,
            'message': 'Loading file...'
        })
        
        # Load and analyze file
        scene = quick_load(file_path, analyze=True)
        scene_id = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}"
        scene.id = scene_id
        
        # Update progress
        processing_jobs[job_id].update({
            'progress': 0.5,
            'message': 'Analyzing scene...'
        })
        
        # Store scene
        completed_scenes[scene_id] = scene
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_scene_metrics(scene)
        
        # Complete job
        processing_jobs[job_id].update({
            'status': 'completed',
            'progress': 1.0,
            'message': 'Scene analysis complete',
            'scene_id': scene_id,
            'metrics': metrics.__dict__
        })
        
    except Exception as e:
        processing_jobs[job_id].update({
            'status': 'error',
            'message': f'Error processing file: {str(e)}',
            'error': str(e)
        })
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

async def run_simulation_async(job_id: str, scene_id: str, duration: float, enable_physics: bool):
    """Run simulation asynchronously."""
    try:
        scene = completed_scenes[scene_id]
        
        # Create simulator
        simulator = Simulator(
            scene=scene,
            enable_physics=enable_physics,
            enable_collisions=True
        )
        
        # Progress callback
        def progress_callback(progress: float):
            processing_jobs[job_id]['progress'] = progress
            processing_jobs[job_id]['message'] = f'Simulating... {progress*100:.1f}%'
        
        # Run simulation
        result = simulator.run(duration=duration, progress_callback=progress_callback)
        
        # Calculate metrics
        calculator = MetricsCalculator()
        sim_metrics = calculator.calculate_simulation_metrics(result)
        
        # Store results
        processing_jobs[job_id].update({
            'status': 'completed',
            'progress': 1.0,
            'message': 'Simulation complete',
            'result': {
                'scene_id': scene_id,
                'simulation_time': result.total_time,
                'wall_clock_time': result.wall_clock_time,
                'frames': len(result.frames),
                'metrics': sim_metrics.__dict__,
                'final_metrics': result.final_metrics
            }
        })
        
    except Exception as e:
        processing_jobs[job_id].update({
            'status': 'error',
            'message': f'Simulation failed: {str(e)}',
            'error': str(e)
        })

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the web server."""
    uvicorn.run(
        "scene2sim.web.app:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    run_server()
