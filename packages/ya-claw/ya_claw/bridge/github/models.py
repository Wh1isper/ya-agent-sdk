from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class GitHubUser(BaseModel):
    model_config = ConfigDict(extra="ignore")

    login: str
    id: int
    type: str | None = None


class GitHubRepository(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    full_name: str
    html_url: str


class GitHubNotificationSubject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    url: str | None = None
    latest_comment_url: str | None = None
    type: str


class GitHubNotification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    unread: bool = True
    reason: str
    updated_at: datetime
    last_read_at: datetime | None = None
    subject: GitHubNotificationSubject
    repository: GitHubRepository


@dataclass(frozen=True)
class GitHubNotificationPage:
    notifications: list[GitHubNotification]
    poll_interval_seconds: int | None = None
    response_date: datetime | None = None


@dataclass(frozen=True)
class GitHubNotificationSource:
    payload: dict[str, Any] | None
    sender: GitHubUser | None
    content_text: str | None


@dataclass(frozen=True)
class GitHubNotificationResource:
    kind: str
    number: int
    chat_id: str
    html_url: str
