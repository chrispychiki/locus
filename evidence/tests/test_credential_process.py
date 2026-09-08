"""The credential helper: locus-r2-credentials' key emission is called by an
~/.aws profile's config line, never by an agent, and must hand any S3 tool the
same keys the evidence package derives in-process — from store/.env, nothing exported.
Its --write-profile (install.sh's call) writes that profile, resolving every
fact from the deployment itself, and always says what it did."""

import hashlib
import json
import urllib.request
from pathlib import Path

import pytest
from locus.evidence.credential_process import main

ACCOUNT = "acc1234567890"
TOKEN = "cf-token-value"


class FakeVerify:
    """Stands in for the Cloudflare token-verify endpoint."""

    def __call__(self, request, timeout=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps({"success": True, "result": {"id": "tokenid42"}}).encode()


def test_the_hook_emits_credential_process_json_from_an_env_file(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(urllib.request, "urlopen", FakeVerify())
    env = tmp_path / ".env"
    env.write_text(
        f"# credentials\nCLOUDFLARE_API_TOKEN={TOKEN}\nCLOUDFLARE_ACCOUNT_ID={ACCOUNT}\n"
    )

    main(["--env", str(env)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["Version"] == 1
    assert payload["AccessKeyId"] == "tokenid42"
    assert payload["SecretAccessKey"] == hashlib.sha256(TOKEN.encode()).hexdigest()
    # The expiry is what makes a long-lived caller re-invoke the hook, so a
    # rotated token propagates; a payload without one is cached forever.
    assert payload["Expiration"].endswith("Z")


def test_without_env_file_the_environment_serves(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(urllib.request, "urlopen", FakeVerify())
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)

    main([])

    assert json.loads(capsys.readouterr().out)["AccessKeyId"] == "tokenid42"


def test_missing_credentials_fail_loud_naming_the_env_flag(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(SystemExit, match="--env"):
        main([])


def _clone(root: Path, account: str | None = ACCOUNT) -> None:
    """A deployment at `root` — the two files --write-profile resolves from."""
    (root / "store").mkdir(parents=True)
    (root / "store" / "wrangler.toml").write_text('name = "w"\n')
    if account is not None:
        (root / "store" / ".env").write_text(f"CLOUDFLARE_ACCOUNT_ID={account}\n")


def test_write_profile_appends_the_block_the_key_emission_answers(
    monkeypatch, tmp_path, capsys
):
    # The round trip is the contract: the written credential_process line must
    # be the invocation the key-emission half answers, pointed into this clone.
    _clone(tmp_path)
    config = tmp_path / "aws" / "config"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    main(["--write-profile"])

    text = config.read_text()
    assert "[profile locus]" in text
    assert f"endpoint_url = https://{ACCOUNT}.r2.cloudflarestorage.com" in text
    assert (
        f"credential_process = {tmp_path}/.venv/bin/locus-r2-credentials --env {tmp_path}/store/.env"
    ) in text
    assert str(config) in capsys.readouterr().out


def test_write_profile_appends_after_existing_profiles_untouched(
    monkeypatch, tmp_path, capsys
):
    _clone(tmp_path)
    config = tmp_path / "config"
    config.write_text("[default]\nregion = us-east-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    main(["--write-profile"])

    text = config.read_text()
    assert text.startswith("[default]\nregion = us-east-1\n")
    assert "[profile locus]" in text


def test_an_existing_locus_profile_is_left_alone_and_named(
    monkeypatch, tmp_path, capsys
):
    # It may be another clone's; the disposition line surfaces where it points
    # so a collision is visible at the moment someone is standing at install.
    _clone(tmp_path)
    config = tmp_path / "config"
    before = "[profile locus]\nregion = auto\ncredential_process = /elsewhere/locus-r2-credentials\n"
    config.write_text(before)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    main(["--write-profile"])

    assert config.read_text() == before
    out = capsys.readouterr().out
    assert "left alone" in out
    assert "/elsewhere/locus-r2-credentials" in out


def test_no_account_id_writes_nothing_and_says_so(monkeypatch, tmp_path, capsys):
    # The standing state for a non-R2 S3 backend — a stated fact, not an
    # error: install.sh must keep succeeding.
    _clone(tmp_path, account=None)
    config = tmp_path / "config"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    main(["--write-profile"])

    assert not config.exists()
    out = capsys.readouterr().out
    assert "not written" in out
    assert "CLOUDFLARE_ACCOUNT_ID" in out


def test_write_profile_outside_a_deployment_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    with pytest.raises(SystemExit, match="wrangler.toml"):
        main(["--write-profile"])
