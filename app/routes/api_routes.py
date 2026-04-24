from aiohttp import web
import json


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["clients"].add(ws)
    addr = request.remote or "unknown"

    # Send history on connect
    try:
        hist_file = request.app["history_file"]
        history_data = []
        if hist_file.is_file() and hist_file.stat().st_size > 0:
            try:
                history_data = json.loads(hist_file.read_text())
            except json.JSONDecodeError:
                pass
        await ws.send_str(json.dumps({"type": "history_load", "data": history_data}))
    except Exception as e:
        pass

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await request.app["manager"].handle_message(
                    ws, addr, msg.data
                )
    finally:
        request.app["clients"].discard(ws)
        for s in request.app["subscriptions"].values():
            s.discard(ws)
        await request.app["manager"].check_and_manage_tasks()
    return ws


async def get_history(request):
    file = request.app["history_file"]
    return web.FileResponse(file) if file.is_file() else web.json_response([])
