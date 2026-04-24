from aiohttp import web
from app.main import create_app
import config

if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host=config.HOST, port=config.PORT)
