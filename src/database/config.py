from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'cervical_lesion_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'cervical_lesion_images')

def get_postgres_connection():
    """Create a connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            **POSTGRES_CONFIG,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        raise

def get_mongo_client():
    """Create a connection to MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        return client[MONGO_DB]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise
