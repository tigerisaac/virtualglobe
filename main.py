# main.py (Modified for Packaging)
import logging
import threading
import webbrowser
import time
from waitress import serve
from flask_app import app # Import your Flask app object

# Configure logging (optional but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('waitress')
log.setLevel(logging.INFO)

# --- Configuration ---
HOST = '127.0.0.1' # Run locally
PORT = 5000        # Choose a port

# --- Function to open the browser ---
def open_browser():
    """Waits a moment then opens the browser to the app."""
    time.sleep(1.5) # Give the server a moment to start
    try:
        webbrowser.open(f"http://{HOST}:{PORT}/")
        logging.info(f"Opened browser to http://{HOST}:{PORT}/")
    except Exception as e:
        logging.error(f"Could not open browser: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    logging.info("Starting application server...")

    # Start the browser opening in a separate thread so it doesn't block the server
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Start the Waitress server (this will block until stopped)
    try:
        logging.info(f"Serving on http://{HOST}:{PORT}")
        serve(app, host=HOST, port=PORT, threads=6) # Use Waitress
    except Exception as e:
        logging.error(f"Server failed to start: {e}", exc_info=True)
    finally:
        logging.info("Server shutting down.")