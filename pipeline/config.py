import os

class Config:
    """Configuration settings for the experiment."""
    # Target file
    SOURCE_FILE: str = "evolve file"
    
    # Training script
    BASH_SCRIPT: str = "bash scripts/train.sh"
    
    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"
    
    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"
    
    # Code pool directory
    CODE_POOL: str = "./pool"
    
    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 3
    
    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 10
    
    # RAG service URL (OpenSearch)
    RAG: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

    #OLD
    # Database URL (MongoDB)
    #DATABASE: str = os.getenv("MONGODB_URI", "mongodb://admin:password123@localhost:27018/myapp")
    
    # Additional service URLs for your infrastructure
    #MONGO_EXPRESS_URL: str = "http://localhost:8081"
    #OPENSEARCH_DASHBOARDS_URL: str = "http://localhost:5601"
    
    #NEW
    # Database URL (MongoDB API - FastAPI wrapper)
    DATABASE: str = os.getenv("DATABASE_API_URL", "http://localhost:8000")
    
    # Direct MongoDB connection (for reference/other uses)
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://admin:password123@localhost:27018/myapp")
    
    # Additional service URLs for your infrastructure
    MONGO_EXPRESS_URL: str = "http://localhost:8081"
    OPENSEARCH_DASHBOARDS_URL: str = "http://localhost:5601"
