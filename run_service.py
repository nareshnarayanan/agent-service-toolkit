from dotenv import load_dotenv

# Load environment variables first before imports.
load_dotenv()

import uvicorn
from service import app
uvicorn.run(app, host="0.0.0.0", port=80, log_level="debug")
