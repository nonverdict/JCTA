from aiohttp import web
import json


async def get_status(request):
    app = request.app
    detector = app.get("detector")
    models = app.get("models")
    normal_source = app.get("normal_source")
    static_source = app.get("static_source")

    status = {
        "yolo_loaded": detector is not None and detector.model is not None,
        "disease_models_loaded": len(models.disease_models) if models else 0,
        "normal_camera_open": normal_source.is_open() if normal_source else False,
        "static_source_open": static_source.is_open() if static_source else False,
        "active_clients": len(app.get("clients", set())),
        "active_streams": list(app.get("active_streams", set())),
        "lethargy_demo_active": app.get("lethargy_demo_active", False),
    }

    return web.json_response(status)
