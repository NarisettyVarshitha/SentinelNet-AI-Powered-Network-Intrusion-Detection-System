import os
import sys
import subprocess
import joblib
import pandas as pd
import numpy as np
import time
import random
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="SentinelNet AI API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_FILE = os.path.join(BASE_DIR, 'alerts.log')
TRAINING_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_model.py')

# Global state for simulation
class SimulationState:
    running = False
    training = False
    thread = None
    recent_logs = []
    stats = {
        "packets_processed": 0,
        "intrusions_detected": 0,
        "last_confidence": 0.0,
        "status": "Idle"
    }

sim_state = SimulationState()

def simulation_worker():
    global sim_state
    sim_state.running = True
    sim_state.status = "Monitoring..."
    
    try:
        # Load the model
        model_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
        if not os.path.exists(model_path):
            sim_state.status = "Failed: Model not trained"
            sim_state.running = False
            return
            
        model = joblib.load(model_path)
        
        # Load seed data
        test_path = os.path.join(DATA_DIR, 'test_selected.csv')
        if not os.path.exists(test_path):
            sim_state.status = "Failed: Test data not found"
            sim_state.running = False
            return
            
        seed_data = pd.read_csv(test_path, nrows=100)
        feature_cols = [c for c in seed_data.columns if c != 'label']
        numeric_feature_cols = seed_data[feature_cols].select_dtypes(include=np.number).columns
        normal_seeds = seed_data[seed_data['label'] == 0].drop('label', axis=1)
        attack_seeds = seed_data[seed_data['label'] == 1].drop('label', axis=1)

        while sim_state.running:
            # Generate a "live" packet
            is_attack = random.random() < 0.20
            if is_attack and not attack_seeds.empty:
                base_packet = attack_seeds.sample(1).iloc[0].copy()
                packet_type = "Suspicious Packet"
            elif not normal_seeds.empty:
                base_packet = normal_seeds.sample(1).iloc[0].copy()
                packet_type = "User Activity"
            else:
                time.sleep(1)
                continue

            # Add some noise
            for col in numeric_feature_cols:
                if base_packet[col] > 1:
                    base_packet[col] *= random.uniform(0.9, 1.1)

            live_packet_df = pd.DataFrame([base_packet], columns=feature_cols)
            prediction = int(model.predict(live_packet_df)[0])
            confidence = float(model.predict_proba(live_packet_df)[0].max())
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "packet_type": packet_type,
                "prediction": "Intrusion" if prediction == 1 else "Normal",
                "confidence": round(confidence, 2),
                "is_alert": prediction == 1
            }
            
            # Update stats
            sim_state.stats["packets_processed"] += 1
            if prediction == 1:
                sim_state.stats["intrusions_detected"] += 1
            sim_state.stats["last_confidence"] = log_entry["confidence"]
            
            # Keep last 50 logs
            sim_state.recent_logs.insert(0, log_entry)
            sim_state.recent_logs = sim_state.recent_logs[:50]
            
            # Write to disk log if it's an intrusion
            if prediction == 1:
                with open(LOG_FILE, 'a') as f:
                    f.write(f"[{timestamp}] ALERT: {packet_type} detected with {confidence:.2f} confidence.\n")

            time.sleep(random.uniform(0.5, 2.0))
            
    except Exception as e:
        print(f"Simulation Error: {e}")
        sim_state.status = f"Error: {str(e)}"
    finally:
        sim_state.running = False

@app.get("/")
def read_root():
    return {"status": "SentinelNet AI API is live"}

@app.get("/stats")
def get_stats():
    # Try to load evaluation stats if they exist
    eval_stats = {}
    # (In a real app, I'd read the output of 06_model_evaluation.py)
    # For now, return the live stats
    return {
        "live": sim_state.stats,
        "is_running": sim_state.running,
        "system_status": sim_state.status
    }

@app.get("/logs")
def get_logs():
    return sim_state.recent_logs

@app.get("/model-info")
def get_model_info():
    model_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    model_exists = os.path.isfile(model_path)
    model_info = {
        "available": model_exists,
        "name": "Random Forest",
        "dataset": "NSL-KDD",
        "input_features": 41,
        "classes": ["Normal", "Intrusion"],
        "artifact_size_bytes": None,
        "last_trained": None,
    }
    if model_exists:
        model_stat = os.stat(model_path)
        model_info["artifact_size_bytes"] = model_stat.st_size
        model_info["last_trained"] = datetime.fromtimestamp(
            model_stat.st_mtime
        ).astimezone().isoformat(timespec="seconds")
    return model_info

@app.get("/train")
def train_instructions():
    return {"message": "Training is started with POST /train. Use the frontend Re-Train AI button or send a POST request."}

@app.post("/simulation/start")
def start_simulation(background_tasks: BackgroundTasks):
    if not sim_state.running:
        sim_state.thread = threading.Thread(target=simulation_worker)
        sim_state.thread.daemon = True
        sim_state.thread.start()
        return {"message": "Simulation started"}
    return {"message": "Simulation already running"}

@app.post("/simulation/stop")
def stop_simulation():
    sim_state.running = False
    sim_state.status = "Stopped"
    return {"message": "Simulation stopping..."}

@app.post("/train")
def run_training(background_tasks: BackgroundTasks):
    if sim_state.training:
        return {"message": "Training is already in progress"}

    sim_state.training = True
    sim_state.status = "Training in progress..."

    def train_task():
        sim_state.status = "Training in progress..."
        try:
            result = subprocess.run(
                [sys.executable, TRAINING_SCRIPT],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=False
            )
            if result.stdout:
                print(result.stdout, end="")
            if result.returncode != 0:
                error = result.stderr.strip() or f"Trainer exited with code {result.returncode}"
                print(f"Training failed:\n{error}")
                sim_state.status = f"Training failed: {error[-300:]}"
                return
            sim_state.status = "Training complete"
        except Exception as e:
            sim_state.status = f"Training failed: {e}"
            print(f"Training failed: {e}")
        finally:
            sim_state.training = False
        
    background_tasks.add_task(train_task)
    return {"message": "Training pipeline started in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
