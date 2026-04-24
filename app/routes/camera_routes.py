from aiohttp import web
import config


async def handle_index(request):
    return web.FileResponse(config.TEMPLATES_DIR / "plan.html")


async def handle_static(request):
    filename = request.match_info.get("filename")
    try:
        file_path = (config.STATIC_DIR / filename).resolve()
    except (ValueError, RuntimeError):
        return web.Response(status=404, text="Asset not found")

    # Security: Ensure the resolved path is actually inside the STATIC_DIR
    try:
        file_path.relative_to(config.STATIC_DIR.resolve())
    except ValueError:
        return web.Response(status=404, text="Asset not found")

    if file_path.is_file():
        return web.FileResponse(file_path)

    return web.Response(status=404, text="Asset not found")
