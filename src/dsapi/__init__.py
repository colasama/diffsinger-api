import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="dsapi")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the whole pipeline once, synchronously")
    run.add_argument("song_id")
    run.add_argument("--start", type=float)
    run.add_argument("--end", type=float)
    run.add_argument("--modes", default="mimic,score")
    run.add_argument("--with-accompaniment", action="store_true", help="also mix synthesized vocals with the original instrumental")
    run.add_argument("--audio-only", action="store_true", help="mimic only: skip note transcription and score output")
    run.add_argument("--task-id")
    run.add_argument("--voice", help="ID from your imported voice catalog")
    run.add_argument("--synthesis-seed", type=int)
    run.add_argument("--lyric-extent", choices=("auto", "window", "exact"), default="auto")
    run.add_argument("--selection", choices=("random", "chorus"), default="random")
    run.add_argument("--seed", type=int, help="reproduce a random lyric window")
    run.add_argument("--lyric-query", help="prefer an excerpt starting at this lyric line")
    ev = sub.add_parser("eval", help="objective quality checks for a finished task")
    ev.add_argument("task_id")
    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--webui", action="store_true", help="also serve the web UI at /ui (or DSAPI_WEBUI=1)")
    args = p.parse_args()

    if args.cmd == "run":
        from .pipeline import Job, JobSpec
        from . import voices

        task_id = args.task_id or f"song{args.song_id}"
        st = Job(task_id, JobSpec(song_id=args.song_id, start=args.start, end=args.end,
                                  modes=args.modes.split(","), selection=args.selection, seed=args.seed,
                                  lyric_query=args.lyric_query, lyric_extent=args.lyric_extent,
                                  voice_id=args.voice or voices.DEFAULT_VOICE, synthesis_seed=args.synthesis_seed,
                                  with_accompaniment=args.with_accompaniment, audio_only=args.audio_only)).run()
        print(st.jianpu or "")
        print(json.dumps({"status": st.status, "error": st.error, "timings": st.timings,
                          "window": st.window, "alignment": st.alignment, "outputs": st.outputs},
                         ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        from . import settings
        from .evaluate import evaluate

        print(json.dumps(evaluate(settings.WORK / args.task_id), indent=2))
    else:
        import os

        import uvicorn

        if args.webui:
            os.environ["DSAPI_WEBUI"] = "1"

        uvicorn.run("dsapi.api:app", host=args.host, port=args.port)
