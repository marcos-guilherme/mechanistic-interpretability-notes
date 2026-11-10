from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download


MODEL = "google/gemma-2b-it"


def main() -> int:
    path = Path.home() / ".cache" / "huggingface" / "token"
    token = get_token()
    print(f"token_file_exists={path.is_file()}")
    print(f"token_detected={token is not None}")
    if token is None:
        return 1
    HfApi().whoami(token=token)
    print("token_valid=True")
    hf_hub_download(MODEL, "config.json", token=token)
    print(f"model_access=True model={MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
