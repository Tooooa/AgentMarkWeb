
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "sk-028c7d27014d4feb892e0d05974f6ff4"

def test_add_agent():
    print(f"Testing Add Agent against {BASE_URL}...")
    
    # 1. Init
    try:
        init_res = requests.post(f"{BASE_URL}/api/add_agent/init", json={
            "modelUrl": "gpt-4o",
            "apiKey": API_KEY
        })
        init_res.raise_for_status()
        session_data = init_res.json()
        session_id = session_data["sessionId"]
        print(f"Session Initialized: {session_id}")
    except Exception as e:
        print(f"Init Failed: {e}")
        try:
             print(init_res.text)
        except: pass
        sys.exit(1)

    # 2. Turn
    try:
        turn_res = requests.post(f"{BASE_URL}/api/add_agent/turn", json={
            "sessionId": session_id,
            "message": "Check the weather in Tokyo",
            "apiKey": API_KEY
        })
        turn_res.raise_for_status()
        turn_data = turn_res.json()
        
        steps = turn_data.get("steps", [])
        trace = turn_data.get("promptTrace")
        
        print(f"Turn Success. Steps: {len(steps)}")
        if trace:
            print("Prompt Trace: FOUND (Length: " + str(len(trace)) + ")")
        else:
            print("Prompt Trace: MISSING")
            
        for s in steps:
            print(f"Step {s['stepIndex']}: Action='{s.get('action')}' | Thought='{s.get('thought')[:50]}...'")
            if s.get('watermark', {}).get('bits'):
                 print(f"  Watermark bits: {s['watermark']['bits']}")
            if s.get('toolDetails'):
                 print(f"  ToolDetails: {s.get('toolDetails')[:50]}...")
            
    except Exception as e:
        print(f"Turn Failed: {e}")
        try:
             print(turn_res.text)
        except: pass
        sys.exit(1)

if __name__ == "__main__":
    test_add_agent()
