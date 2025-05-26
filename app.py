from flask import Flask
from routes import bp as routes_bp

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.register_blueprint(routes_bp)

if __name__ == '__main__':
    app.run(debug=True)