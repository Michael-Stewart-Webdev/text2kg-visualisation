from flask import Flask

app = Flask(__name__)

from app import routes

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)