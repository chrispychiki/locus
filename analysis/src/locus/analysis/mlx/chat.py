"""One-shot probe of the local MLX server: send a prompt, print the reply.

The hand check that the server answers, that an image lands, that a schema comes back — one request, one exit. It is not how analysis calls the server: it sends no sampling, so every omitted field falls to upstream's own schema defaults, which makes it a view of raw upstream behavior. The model is whatever the server has loaded (asked once via /health) and stated explicitly in the request. Image paths anywhere in the prompt are sent as images (client and server share a filesystem). Target is the card's endpoint (sole_bind from the analysis card's base_url); LOCUS_MLX_URL overrides.

Usage:
    locus-mlx-chat --prompt "what is in /abs/path/image.png?"
    locus-mlx-chat --prompt "count the vowels in banana" --json-schema '{"type":"object","properties":{"vowels":{"type":"integer"}}}'
    locus-mlx-chat --prompt "say hello" --no-think --verbose
"""

import argparse
import json
import os
import sys
import urllib.request

from .serve import sole_bind

_host, _port = sole_bind()
SERVER_URL = os.environ.get("LOCUS_MLX_URL", f"http://{_host}:{_port}")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".heif"}

# The server is local by construction, so no proxy configured in the environment
# may sit between this probe and it.
_direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_loaded_model: str | None = None


def loaded_model() -> str:
    global _loaded_model
    if _loaded_model is None:
        with _direct.open(f"{SERVER_URL}/health", timeout=5) as response:
            model = json.loads(response.read()).get("loaded_model")
        if not model:
            raise RuntimeError(
                "server is up but reports no loaded model — boot it via "
                "locus-mlx-serve, which preloads the variant's repo"
            )
        _loaded_model = model
    return _loaded_model


def image_paths(text: str) -> list[str]:
    return [
        word
        for word in text.split()
        if os.path.splitext(word)[1].lower() in IMAGE_EXTENSIONS
        and os.path.isfile(word)
    ]


def _load_schema(arg: str) -> dict:
    if os.path.isfile(arg):
        with open(arg) as f:
            return json.load(f)
    return json.loads(arg)


def main() -> None:
    import openai

    parser = argparse.ArgumentParser(
        description="Send one prompt to the local MLX server (OpenAI /v1/chat/completions) and print the reply."
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="The prompt. Image paths in it are sent as images.",
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.", help="System prompt"
    )
    parser.add_argument(
        "--no-think", "-n", action="store_true", help="Disable thinking mode"
    )
    parser.add_argument(
        "--json-schema",
        default=None,
        help="Inline JSON schema string OR path to a .json file; sets response_format=json_schema",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the whole response body (usage, timings) instead of the reply",
    )
    args = parser.parse_args()

    content: list = [
        {"type": "image_url", "image_url": {"url": os.path.abspath(path)}}
        for path in image_paths(args.prompt)
    ]
    content.append({"type": "text", "text": args.prompt})
    params: dict = {
        "model": loaded_model(),
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "extra_body": {"enable_thinking": not args.no_think},
        "timeout": 21600,
    }
    if args.json_schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": _load_schema(args.json_schema)},
        }

    client = openai.OpenAI(
        base_url=f"{SERVER_URL}/v1",
        api_key="local",
        http_client=openai.DefaultHttpxClient(trust_env=False),
    )
    try:
        raw = client.chat.completions.with_raw_response.create(**params)
    except openai.APIConnectionError:
        print(
            f"No server at {SERVER_URL} — boot it with locus-mlx-serve.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.verbose:
        print(json.dumps(raw.http_response.json(), indent=2))
        return
    message = raw.parse().choices[0].message
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        print(f"[reasoning] {reasoning}\n")
    print(message.content or "")


if __name__ == "__main__":
    main()
