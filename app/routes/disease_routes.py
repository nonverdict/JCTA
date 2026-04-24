from aiohttp import web

# INFO: for further diseaseData dictionary logic


async def get_disease_info(request):
    return web.json_response({"status": "ok"})
