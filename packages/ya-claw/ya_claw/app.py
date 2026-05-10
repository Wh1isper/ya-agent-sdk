from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine

from ya_claw.api.bridges import router as bridges_router
from ya_claw.api.claw import router as claw_router
from ya_claw.api.health import router as health_router
from ya_claw.api.heartbeat import router as heartbeat_router
from ya_claw.api.profiles import router as profiles_router
from ya_claw.api.runs import router as runs_router
from ya_claw.api.schedules import router as schedules_router
from ya_claw.api.sessions import router as sessions_router
from ya_claw.bridge import BridgeDispatchMode
from ya_claw.bridge.service import BridgeSupervisor, build_bridge_supervisor
from ya_claw.config import ClawSettings, get_settings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution import (
    ClawRuntimeBuilder,
    ExecutionSupervisor,
    ProfileResolver,
    RunDispatcher,
    RuntimeInstanceManager,
    SessionPruneDispatcher,
)
from ya_claw.execution.heartbeat import HeartbeatDispatcher
from ya_claw.execution.schedule import ScheduleDispatcher
from ya_claw.logging import configure_claw_logging, redact_url
from ya_claw.notifications import NotificationHub, create_notification_hub
from ya_claw.runtime_state import InMemoryRuntimeState, create_runtime_state
from ya_claw.workspace import (
    DefaultEnvironmentFactory,
    DockerWorkspaceProvider,
    LocalWorkspaceProvider,
    WorkspaceProvider,
)


class ClawApplication:
    reserved_frontend_paths = ("api", "docs", "redoc", "openapi.json", "healthz")
    auth_exempt_paths = frozenset({"/healthz", "/docs", "/redoc", "/openapi.json"})

    def __init__(self, settings: ClawSettings):
        self.settings = settings

    def create(self) -> FastAPI:
        configure_claw_logging(self.settings.log_level)
        logger.info("Creating YA Claw FastAPI app log_level={}", self.settings.log_level)
        self.settings.require_api_token()

        app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.resolved_service_revision,
            description="Workspace-native single-node agent runtime with in-process state and SQLite-first storage.",
            lifespan=self.lifespan,
        )
        app.state.settings = self.settings
        app.state.db_engine = None
        app.state.db_session_factory = None
        app.state.runtime_state = None
        app.state.notification_hub = None
        app.state.workspace_provider = None
        app.state.environment_factory = None
        app.state.profile_resolver = None
        app.state.runtime_builder = None
        app.state.execution_supervisor = None
        app.state.runtime_instance_manager = None
        app.state.bridge_supervisor = None
        app.state.schedule_dispatcher = None
        app.state.heartbeat_dispatcher = None
        app.state.session_prune_dispatcher = None

        self.register_api_token_middleware(app)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(health_router)
        app.include_router(claw_router, prefix="/api/v1")
        app.include_router(profiles_router, prefix="/api/v1")
        app.include_router(sessions_router, prefix="/api/v1")
        app.include_router(runs_router, prefix="/api/v1")
        app.include_router(schedules_router, prefix="/api/v1")
        app.include_router(heartbeat_router, prefix="/api/v1")
        app.include_router(bridges_router, prefix="/api/v1")

        frontend_registered = self.register_frontend(app)
        if not frontend_registered:
            self.register_index(app)

        return app

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:  # noqa: C901
        logger.info(
            "Starting YA Claw app environment={} instance_id={} host={} port={} public_base_url={}",
            self.settings.environment,
            self.settings.instance_id,
            self.settings.host,
            self.settings.port,
            self.settings.public_base_url,
        )
        self.settings.ensure_runtime_directories()
        logger.info(
            "Runtime directories ready data_dir={} run_store_dir={} workspace_dir={}",
            self.settings.runtime_data_dir,
            self.settings.run_store_dir,
            self.settings.resolved_workspace_dir,
        )

        app.state.db_engine = create_engine(
            self.settings.resolved_database_url,
            echo=self.settings.database_echo,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow,
            pool_recycle=self.settings.database_pool_recycle_seconds,
        )
        app.state.db_session_factory = create_session_factory(app.state.db_engine)
        logger.info("Database engine ready url={}", redact_url(self.settings.resolved_database_url))
        app.state.runtime_state = create_runtime_state()
        app.state.notification_hub = create_notification_hub()
        app.state.workspace_provider = self.create_workspace_provider()
        logger.info("Workspace provider ready backend={}", self.settings.workspace_provider_backend)
        app.state.environment_factory = DefaultEnvironmentFactory(
            docker_image=self.settings.workspace_provider_docker_image,
            workspace_uid=self.settings.resolved_workspace_provider_docker_uid,
            workspace_gid=self.settings.resolved_workspace_provider_docker_gid,
            workspace_environment=self.settings.resolved_workspace_environment,
            docker_container_cache_dir=self.settings.resolved_workspace_provider_docker_container_cache_dir,
            docker_extra_mounts=self.settings.resolved_workspace_provider_docker_extra_mounts,
            docker_exec_user=self.settings.resolved_workspace_provider_docker_exec_user,
            docker_exec_default_env=self.settings.resolved_workspace_provider_docker_exec_default_env,
        )

        if app.state.db_session_factory is not None:
            app.state.profile_resolver = ProfileResolver(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
            )
            if self.settings.auto_seed_profiles:
                seeded_profiles = await app.state.profile_resolver.seed_profiles()
                logger.info("Profile auto-seed completed count={} names={}", len(seeded_profiles), seeded_profiles)
            app.state.runtime_builder = ClawRuntimeBuilder(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
            )

        if (
            isinstance(app.state.runtime_state, InMemoryRuntimeState)
            and app.state.db_session_factory is not None
            and app.state.profile_resolver is not None
            and app.state.runtime_builder is not None
        ):
            supervisor = ExecutionSupervisor(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
                runtime_state=app.state.runtime_state,
                workspace_provider=app.state.workspace_provider,
                environment_factory=app.state.environment_factory,
                profile_resolver=app.state.profile_resolver,
                runtime_builder=app.state.runtime_builder,
                notification_hub=app.state.notification_hub,
            )
            app.state.execution_supervisor = supervisor
            app.state.runtime_instance_manager = RuntimeInstanceManager(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
            )
            await app.state.runtime_instance_manager.register(metadata={"environment": self.settings.environment})
            logger.info("Runtime instance registered instance_id={}", self.settings.instance_id)
            recovery_result = await supervisor.startup_recover()
            logger.info("Execution supervisor startup recovery completed result={}", recovery_result)
            schedule_dispatcher = ScheduleDispatcher(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
                runtime_state=app.state.runtime_state,
                run_dispatcher=RunDispatcher(supervisor),
                notification_hub=app.state.notification_hub,
            )
            app.state.schedule_dispatcher = schedule_dispatcher
            await schedule_dispatcher.startup()
            heartbeat_dispatcher = HeartbeatDispatcher(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
                runtime_state=app.state.runtime_state,
                run_dispatcher=RunDispatcher(supervisor),
                notification_hub=app.state.notification_hub,
            )
            app.state.heartbeat_dispatcher = heartbeat_dispatcher
            await heartbeat_dispatcher.startup()
            session_prune_dispatcher = SessionPruneDispatcher(
                settings=self.settings,
                session_factory=app.state.db_session_factory,
            )
            app.state.session_prune_dispatcher = session_prune_dispatcher
            await session_prune_dispatcher.startup()
            if self.settings.bridge_dispatch_mode == BridgeDispatchMode.EMBEDDED:
                bridge_supervisor = build_bridge_supervisor(
                    settings=self.settings,
                    session_factory=app.state.db_session_factory,
                    runtime_state=app.state.runtime_state,
                    run_dispatcher=RunDispatcher(supervisor),
                    notification_hub=app.state.notification_hub,
                )
                app.state.bridge_supervisor = bridge_supervisor
                await bridge_supervisor.startup()
                logger.info(
                    "Bridge supervisor started adapters={}", sorted(self.settings.resolved_bridge_enabled_adapters)
                )

        logger.info("YA Claw app startup complete")
        try:
            yield
        finally:
            logger.info("Shutting down YA Claw app")
            db_engine = app.state.db_engine
            runtime_state = app.state.runtime_state
            notification_hub = app.state.notification_hub
            bridge_supervisor = app.state.bridge_supervisor
            schedule_dispatcher = app.state.schedule_dispatcher
            heartbeat_dispatcher = app.state.heartbeat_dispatcher
            session_prune_dispatcher = app.state.session_prune_dispatcher
            execution_supervisor = app.state.execution_supervisor

            app.state.db_session_factory = None
            app.state.workspace_provider = None
            app.state.environment_factory = None
            app.state.profile_resolver = None
            app.state.mcp_config_resolver = None
            app.state.runtime_builder = None
            runtime_instance_manager = app.state.runtime_instance_manager
            app.state.execution_supervisor = None
            app.state.runtime_instance_manager = None
            app.state.bridge_supervisor = None
            app.state.schedule_dispatcher = None
            app.state.heartbeat_dispatcher = None
            app.state.session_prune_dispatcher = None

            if isinstance(session_prune_dispatcher, SessionPruneDispatcher):
                await session_prune_dispatcher.shutdown()

            if isinstance(heartbeat_dispatcher, HeartbeatDispatcher):
                await heartbeat_dispatcher.shutdown()

            if isinstance(schedule_dispatcher, ScheduleDispatcher):
                await schedule_dispatcher.shutdown()

            if isinstance(bridge_supervisor, BridgeSupervisor):
                await bridge_supervisor.shutdown()

            if isinstance(execution_supervisor, ExecutionSupervisor):
                await execution_supervisor.shutdown()

            if isinstance(runtime_instance_manager, RuntimeInstanceManager):
                await runtime_instance_manager.mark_stopped()

            if isinstance(runtime_state, InMemoryRuntimeState):
                await runtime_state.aclose()

            if isinstance(notification_hub, NotificationHub):
                await notification_hub.aclose()

            if isinstance(db_engine, AsyncEngine):
                await db_engine.dispose()

    def create_workspace_provider(self) -> WorkspaceProvider:
        if self.settings.workspace_provider_backend == "docker":
            logger.info(
                "Configuring Docker workspace provider image={} service_workspace_dir={} docker_host_workspace_dir={} extra_mounts={} exec_user={}",
                self.settings.workspace_provider_docker_image,
                self.settings.resolved_workspace_dir,
                self.settings.resolved_workspace_provider_docker_host_workspace_dir,
                [
                    (str(mount.host_path), str(mount.container_path), mount.mode)
                    for mount in self.settings.resolved_workspace_provider_docker_extra_mounts
                ],
                self.settings.resolved_workspace_provider_docker_exec_user,
            )
            return DockerWorkspaceProvider(
                self.settings.resolved_workspace_dir,
                image=self.settings.workspace_provider_docker_image,
                docker_host_workspace_dir=self.settings.resolved_workspace_provider_docker_host_workspace_dir,
                extra_mounts=self.settings.resolved_workspace_provider_docker_extra_mounts,
            )
        logger.info("Configuring local workspace provider workspace_dir={}", self.settings.resolved_workspace_dir)
        return LocalWorkspaceProvider(self.settings.resolved_workspace_dir)

    def register_api_token_middleware(self, app: FastAPI) -> None:
        expected_token = self.settings.require_api_token()

        @app.middleware("http")
        async def require_api_token(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            if (
                request.method == "OPTIONS"
                or request.url.path in self.auth_exempt_paths
                or self.is_public_frontend_path(request.url.path)
            ):
                return await call_next(request)

            authorization_header = request.headers.get("Authorization")
            provided_token = self.resolve_bearer_token(authorization_header)
            if not isinstance(provided_token, str) or not compare_digest(provided_token, expected_token):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={"detail": "Bearer token required."},
                )

            return await call_next(request)

    def resolve_bearer_token(self, authorization_header: str | None) -> str | None:
        if authorization_header is None:
            return None

        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer":
            return None

        normalized_token = token.strip()
        return normalized_token or None

    def is_public_frontend_path(self, request_path: str) -> bool:
        web_dist_dir = self.settings.web_dist_dir
        if web_dist_dir is None or not (web_dist_dir / "index.html").exists():
            return False

        normalized_path = request_path.strip("/")
        return normalized_path not in self.reserved_frontend_paths and not any(
            normalized_path.startswith(f"{prefix}/") for prefix in self.reserved_frontend_paths if "/" not in prefix
        )

    def register_frontend(self, app: FastAPI) -> bool:
        web_dist_dir = self.settings.web_dist_dir
        if web_dist_dir is None:
            return False

        index_file = web_dist_dir / "index.html"
        if not web_dist_dir.exists() or not index_file.exists():
            return False

        @app.get("/", include_in_schema=False)
        async def frontend_index() -> FileResponse:
            return FileResponse(index_file)

        @app.get("/{full_path:path}", include_in_schema=False)
        async def frontend_route(full_path: str) -> FileResponse:
            normalized_path = full_path.strip("/")
            if normalized_path in self.reserved_frontend_paths or any(
                normalized_path.startswith(f"{prefix}/") for prefix in self.reserved_frontend_paths if "/" not in prefix
            ):
                raise HTTPException(status_code=404, detail="Route not found.")

            return FileResponse(self.resolve_frontend_target(web_dist_dir, normalized_path))

        return True

    def resolve_frontend_target(self, web_dist_dir: Path, requested_path: str) -> Path:
        relative_path = requested_path.strip("/")
        if relative_path == "":
            return web_dist_dir / "index.html"

        candidate = (web_dist_dir / relative_path).resolve()
        web_root = web_dist_dir.resolve()

        try:
            candidate.relative_to(web_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Invalid frontend asset path.") from exc

        if candidate.is_file():
            return candidate

        return web_dist_dir / "index.html"

    def register_index(self, app: FastAPI) -> None:
        @app.get("/")
        async def index() -> dict[str, object]:
            return {
                "name": self.settings.app_name,
                "environment": self.settings.environment,
                "docs_url": "/docs",
                "spec_path": "packages/ya-claw/spec",
                "surfaces": ["profiles", "sessions", "runs", "schedules", "bridges"],
            }


def create_app() -> FastAPI:
    settings = get_settings()
    configure_claw_logging(settings.log_level)
    logger.info("create_app loaded settings environment={} log_level={}", settings.environment, settings.log_level)
    return ClawApplication(settings).create()
