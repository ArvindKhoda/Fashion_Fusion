import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import gridfs

# MongoDB connection URI
uri = "mongodb+srv://arvindkhoda:7725920259@cluster0.0ss4i.mongodb.net/?retryWrites=true&w=majority&tls=true"

client = MongoClient(uri, tlsCAFile=certifi.where())
# Use consistent database name
db = client["Fashion_Fusion"]

# Collections and GridFS
fs = gridfs.GridFS(db)                          # For storing image files
user_images_collection = db["user_images"]      # For storing image metadata or URLs
user = db["users"]                  # For user account data

# Test MongoDB connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Connection error:", e)
