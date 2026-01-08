import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dashboard.server.retriever import ToolBenchRetriever
except ImportError:
    # Just in case run from wrong dir
    from server.retriever import ToolBenchRetriever

def test_manual():
    data_root = PROJECT_ROOT / "experiments/toolbench/data/data/toolenv/tools"
    print(f"Testing with data root: {data_root}")
    
    if not data_root.exists():
        print("[ERROR] Data root not found!")
        return

    retriever = ToolBenchRetriever(str(data_root))
    retriever.load_model()
    retriever.index_tools()
    
    # Test queries
    queries = [
        "I need to check the weather in Paris",
        "Book a hotel in New York",
        "Get stock price for Apple"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        if not results:
            print("  [WARN] No results found.")
        
        for res in results:
            print(f"  - [{res.get('category_name')}] {res.get('tool_name')}/{res.get('api_name')}: {res.get('api_description', '')[:50]}...")

if __name__ == "__main__":
    test_manual()
