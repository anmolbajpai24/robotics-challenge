# Dashboard frontend: integration notes

Built and verified 2026-08-02. React + Vite, two pages: Leaderboard, Episodes.
The frozen tables are embedded and render with the API down. The live panel
and the episode player come from the API.

## Where this goes

Unzip into the repo so the layout is:

    dashboard/
      main.py            (existing, untouched)
      frontend/          (this folder: package.json, vite.config.js, index.html, src/)

## Dev loop (two terminals)

    # terminal 1, venv active
    uvicorn dashboard.main:app --port 8011

    # terminal 2
    cd dashboard/frontend
    npm install
    npm run dev          # Vite dev server, proxies /summary /runs /videos /health to 8011

## Production (one process)

    cd dashboard/frontend && npm run build     # emits dashboard/static/

Then in dashboard/main.py, AFTER every existing route (route order matters in
this repo, see Day 10):

    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="dashboard/static", html=True), name="ui")

Verify /summary and /runs still answer after mounting.

## The /videos endpoint (to add in main.py)

The player expects:

    GET /videos            -> [ { "id": "...", "title": "...", "url": "/videos/file/<name>" } ]
    GET /videos/file/<name> -> the mp4, FileResponse

Reference implementation (adapt paths to the real evidence dirs, read-only):

    import os
    from fastapi.responses import FileResponse

    VIDEO_DIRS = ["day16", "outputs/eval"]   # adjust; never write to these

    def _find_videos():
        out = []
        for d in VIDEO_DIRS:
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.endswith(".mp4"):
                        out.append({"id": f, "title": f, "url": f"/videos/file/{f}"})
        return out

    @app.get("/videos")
    def videos():
        return _find_videos()

    @app.get("/videos/file/{name}")
    def video_file(name: str):
        for v in _find_videos():
            if v["id"] == name:
                # resolve real path the same way _find_videos found it
                ...
        # return FileResponse(path, media_type="video/mp4")

    NOTE: define these BEFORE the static mount and any catch-all.

## If /summary renders as raw JSON

The normalizer in src/api.js guesses common field names (policy/name,
success_rate/success/value, arrays under frozen/table/rows/runs). If the real
/summary shape differs, the UI shows the raw payload plus a banner. Fix is a
one-line mapping in normalizeSummary(). Do not change the API to fit the UI.

## Numbers policy

The embedded frozen tables are committed facts (PushT 61.0 / 39.8 / 0.8,
xarm 0.0 / 4.2 / 99.0, oracle 20/20 labeled as privileged). If any number is
ever re-measured, update the embedded copy AND the API together, same commit.
