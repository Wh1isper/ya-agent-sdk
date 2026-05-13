from __future__ import annotations

import json

from ya_oauth.store import OAuthStore
from ya_oauth.types import AuthFile


def test_load_repairs_existing_auth_file_mode(tmp_path) -> None:
    auth_path = tmp_path / ".yaai" / "auth.json"
    auth_path.parent.mkdir(mode=0o700)
    auth_path.write_text(json.dumps(AuthFile().model_dump(mode="json")), encoding="utf-8")
    auth_path.chmod(0o644)

    store = OAuthStore(auth_path)

    assert store.load().version == 1
    assert auth_path.stat().st_mode & 0o777 == 0o600
