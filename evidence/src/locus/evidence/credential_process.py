"""locus-r2-credentials — a credential helper on the standard AWS chain. Two duties, one file, so the profile text and the program it points at cannot drift apart: emit keys (the default — a profile's `credential_process` line names this executable, and any AWS-chain tool resolving that profile shells out to it and reads keys off its stdout), and write that profile (`--write-profile`, install.sh's call).

An `~/.aws` profile wires S3 tooling outside Locus (aws, rclone, anything speaking the standard AWS chain) to the deployment's bucket by shelling out here for keys, derived on the spot from the one Cloudflare token in store/.env. Nothing is exported by hand and nothing derived is persisted, so a rotated token needs no other change. uninstall.sh removes the profile.

The profile, with the deployment's paths filled in (the endpoint host is the CLOUDFLARE_ACCOUNT_ID from that same store/.env):

    [profile locus]
    region = auto
    endpoint_url = https://<account-id>.r2.cloudflarestorage.com
    credential_process = <clone>/.venv/bin/locus-r2-credentials --env <clone>/store/.env

Locus's own reads never call this; they derive the same keys in-process.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from .clock import utc_stamp
from .cloudflare import r2_endpoint, r2_s3_credentials


def _write_profile() -> None:
    """Append the profile to the AWS config, resolving every fact from the deployment itself (the deployment root from cwd upward, the account id from store/.env). Always prints one disposition line — written, already present (never touched: it may be another clone's), or nothing to write because no account id is declared (the standing state for a non-R2 S3 backend, not an error) — and exits clean in every case: install.sh must keep working whatever the disposition."""
    from .deployment import deployment_root, env_from_files

    root = deployment_root()
    config_path = Path(
        os.environ.get("AWS_CONFIG_FILE") or Path.home() / ".aws" / "config"
    )
    existing = config_path.read_text() if config_path.exists() else ""
    block = re.search(
        r"^\[profile locus\][ \t]*$(.*?)(?=^\[|\Z)", existing, re.MULTILINE | re.DOTALL
    )
    if block:
        pointer = re.search(
            r"^credential_process\s*=\s*(.+)$", block.group(1), re.MULTILINE
        )
        points = (
            f"credential_process = {pointer.group(1).strip()}"
            if pointer
            else "no credential_process line"
        )
        print(f"aws profile 'locus' already in {config_path}, left alone ({points})")
        return
    account = env_from_files().get("CLOUDFLARE_ACCOUNT_ID")
    if not account:
        print(
            "aws profile 'locus' not written: no CLOUDFLARE_ACCOUNT_ID "
            "declared in store/.env — a later ./install.sh run writes it "
            "once one exists; a non-R2 S3 backend never declares one and "
            "never needs the profile"
        )
        return
    stanza = (
        f"\n[profile locus]\n"
        f"region = auto\n"
        f"endpoint_url = {r2_endpoint(account)}\n"
        f"credential_process = {root}/.venv/bin/"
        f"locus-r2-credentials --env {root}/store/.env\n"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("a") as handle:
        handle.write(stanza)
    print(f"aws profile 'locus' -> {config_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="locus-r2-credentials",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        help="path to the .env holding CLOUDFLARE_API_TOKEN/"
        "CLOUDFLARE_ACCOUNT_ID — the deployment's store/.env. The caller "
        "is a config file running from an arbitrary cwd, so the profile "
        "passes this absolute; without it, the variables must already be "
        "in the environment",
    )
    parser.add_argument(
        "--write-profile",
        action="store_true",
        help="write the [profile locus] block into $AWS_CONFIG_FILE (default "
        "~/.aws/config) instead of emitting keys, resolving the account "
        "id and clone paths from the deployment at cwd; an existing "
        "profile is never touched. install.sh's call",
    )
    args = parser.parse_args(argv)
    if args.write_profile:
        _write_profile()
        return
    if args.env:
        from .deployment import parse_env_file

        declared = parse_env_file(Path(args.env))
        token = declared.get("CLOUDFLARE_API_TOKEN")
        account = declared.get("CLOUDFLARE_ACCOUNT_ID")
    else:
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        raise SystemExit(
            "CLOUDFLARE_API_TOKEN/CLOUDFLARE_ACCOUNT_ID not found — pass --env <path to store/.env>"
        )
    creds = r2_s3_credentials(account, token)
    # The hour expiry makes the caller re-invoke, so a rotated token
    # propagates into long-lived tools without restarting anything.
    expiry = utc_stamp(int(time.time() * 1000) + 3_600_000)
    print(
        json.dumps(
            {
                "Version": 1,
                "AccessKeyId": creds["access_key_id"],
                "SecretAccessKey": creds["secret_access_key"],
                "Expiration": expiry,
            }
        )
    )
