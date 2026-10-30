"""JSON-lines worker host. Keep vendor imports and model caches in this process."""

import importlib
import json
import os
import sys
import traceback


def serve(module_name: str) -> None:
    # Reserve the original stdout pipe for replies; even native library output
    # must go to stderr so it cannot corrupt the protocol.
    reply = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    module = None
    for line in sys.stdin:
        request = json.loads(line)
        saved_err = os.dup(2)
        try:
            with open(request["log"], "a", buffering=1) as log:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                try:
                    if module is None:
                        module = importlib.import_module(module_name)
                        module.torch.set_grad_enabled(False)
                    module.main(*request["args"])
                    info = module.load_models.cache_info()
                    result = {"ok": True, "pid": os.getpid(),
                              "cache_entries": info.currsize, "cache_loads": info.misses}
                    if hasattr(module, "runtime_info"):
                        result["runtime"] = module.runtime_info()
                except Exception as exc:
                    traceback.print_exc()
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
        finally:
            os.dup2(saved_err, 1)
            os.dup2(saved_err, 2)
            os.close(saved_err)
        reply.write(json.dumps(result) + "\n")
    reply.close()


if __name__ == "__main__":
    serve(sys.argv[1])
