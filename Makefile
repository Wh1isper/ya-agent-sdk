YA_CLAW_SERVICE_VERSION ?= $(shell uv run python -c "from importlib.metadata import version; print(version('ya-claw'))")
YA_CLAW_SERVICE_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || true)
YA_CLAW_SERVICE_BUILD ?= dev
YA_CLAW_SERVICE_IMAGE ?= ya-claw:dev

.PHONY: install
install: ## Install Python, web dependencies, and pre-commit hooks
	@echo "Creating workspace environment using uv"
	@uv sync --all-packages
	@echo "Installing web dependencies with corepack pnpm"
	@corepack pnpm install
	@uv run pre-commit install

.PHONY: install-skills
install-skills: ## Install canonical skills into ~/.agents/skills
	@echo "Installing skills into $$HOME/.agents/skills"
	@rm -rf "$$HOME/.agents/skills/agent-builder"
	@mkdir -p "$$HOME/.agents/skills/agent-builder"
	@cp -R skills/agent-builder/. "$$HOME/.agents/skills/agent-builder/"
	@mkdir -p "$$HOME/.agents/skills/agent-builder/examples"
	@cp -R examples/* "$$HOME/.agents/skills/agent-builder/examples/"
	@cp examples/.env.example "$$HOME/.agents/skills/agent-builder/examples/"
	@rm -rf "$$HOME/.agents/skills/ya-claw-deploy"
	@mkdir -p "$$HOME/.agents/skills/ya-claw-deploy"
	@cp -R skills/ya-claw-deploy/. "$$HOME/.agents/skills/ya-claw-deploy/"

.PHONY: lint
lint: ## Lint the code
	@echo "Checking lock file consistency with pyproject.toml"
	@uv lock --locked
	@echo "Running pre-commit"
	@uv run pre-commit run -a

.PHONY: cli
cli: ## Run the CLI
	@echo "Running yaacli"
	@./scripts/sync-skills.sh
	@rm -f yaacli.log && YAACLI_PERF=1 uv run --package yaacli yaacli -v

.PHONY: run-claw
run-claw: ## Run the YA Claw backend locally
	@echo "Running ya-claw"
	@uv run --package ya-claw ya-claw serve --reload

.PHONY: claw-db-upgrade
claw-db-upgrade: ## Run YA Claw DB migrations to latest
	@echo "Upgrading ya-claw database"
	@uv run --package ya-claw ya-claw db upgrade

.PHONY: claw-db-downgrade
claw-db-downgrade: ## Roll back YA Claw DB by one migration
	@echo "Downgrading ya-claw database"
	@uv run --package ya-claw ya-claw db downgrade

.PHONY: claw-db-current
claw-db-current: ## Show current YA Claw DB revision
	@echo "Showing ya-claw database revision"
	@uv run --package ya-claw ya-claw db current

.PHONY: claw-db-history
claw-db-history: ## Show YA Claw migration history
	@echo "Showing ya-claw migration history"
	@uv run --package ya-claw ya-claw db history

.PHONY: claw-db-migrate
claw-db-migrate: ## Generate a YA Claw migration (MSG required)
	@echo "Generating ya-claw migration"
	@uv run --package ya-claw ya-claw db migrate "$(MSG)"

.PHONY: web-install
web-install: ## Install web app dependencies with corepack pnpm
	@echo "Installing ya-claw-web dependencies"
	@corepack pnpm install

.PHONY: web-dev
web-dev: ## Run the YA Claw web app locally
	@echo "Running ya-claw-web"
	@corepack pnpm --dir apps/ya-claw-web dev

.PHONY: web-lint
web-lint: ## Run ESLint for the YA Claw web app
	@echo "Running ya-claw-web lint"
	@corepack pnpm --dir apps/ya-claw-web exec eslint .

.PHONY: web-build
web-build: ## Run TypeScript and Vite build checks for the YA Claw web app
	@echo "Running ya-claw-web build"
	@corepack pnpm --dir apps/ya-claw-web build

.PHONY: desktop-dev
desktop-dev: ## Run the YA Desktop Tauri app locally
	@echo "Running ya-desktop"
	@corepack pnpm --dir apps/ya-desktop tauri:dev

.PHONY: desktop-lint
desktop-lint: ## Run ESLint for the YA Desktop app
	@echo "Running ya-desktop lint"
	@corepack pnpm --dir apps/ya-desktop exec eslint .

.PHONY: desktop-build
desktop-build: ## Run TypeScript and Vite build checks for the YA Desktop app
	@echo "Running ya-desktop build"
	@corepack pnpm --dir apps/ya-desktop build

.PHONY: desktop-test
desktop-test: ## Run YA Desktop frontend tests
	@echo "Running ya-desktop tests"
	@corepack pnpm --dir apps/ya-desktop test

.PHONY: desktop-rust-fmt
desktop-rust-fmt: ## Check YA Desktop Rust formatting
	@echo "Checking ya-desktop Rust formatting"
	@(cd apps/ya-desktop/src-tauri && cargo fmt --all -- --check)

.PHONY: desktop-rust-check
desktop-rust-check: ## Run cargo check for the YA Desktop Tauri app
	@echo "Running ya-desktop Rust check"
	@(cd apps/ya-desktop/src-tauri && cargo check --locked --all-targets)

.PHONY: desktop-rust-clippy
desktop-rust-clippy: ## Run clippy for the YA Desktop Tauri app
	@echo "Running ya-desktop Rust clippy"
	@(cd apps/ya-desktop/src-tauri && cargo clippy --locked --all-targets --all-features -- -D warnings)

.PHONY: desktop-tauri-build
desktop-tauri-build: ## Build the YA Desktop Tauri bundle
	@echo "Building ya-desktop Tauri bundle"
	@corepack pnpm --dir apps/ya-desktop tauri:build

.PHONY: docker-build-claw
docker-build-claw: ## Build the YA Claw Docker image
	@echo "Building ya-claw Docker image"
	@docker build \
		--build-arg YA_CLAW_SERVICE_VERSION="$(YA_CLAW_SERVICE_VERSION)" \
		--build-arg YA_CLAW_SERVICE_COMMIT="$(YA_CLAW_SERVICE_COMMIT)" \
		--build-arg YA_CLAW_SERVICE_BUILD="$(YA_CLAW_SERVICE_BUILD)" \
		--build-arg YA_CLAW_SERVICE_IMAGE="$(YA_CLAW_SERVICE_IMAGE)" \
		-f Dockerfile.ya-claw -t "$(YA_CLAW_SERVICE_IMAGE)" .

.PHONY: docker-build-claw-workspace
docker-build-claw-workspace: ## Build the YA Claw workspace Docker image
	@echo "Building ya-claw workspace Docker image"
	@docker build -f Dockerfile.ya-claw-workspace -t ya-claw-workspace:dev .

.PHONY: docker-run-claw
docker-run-claw: ## Run the YA Claw Docker image
	@echo "Running ya-claw Docker image"
	@docker run --rm -p 9042:9042 ya-claw:dev

.PHONY: docker-build-platform
docker-build-platform: ## Build the YA Agent Platform Docker image
	@echo "Building ya-agent-platform Docker image"
	@docker build -f Dockerfile.ya-agent-platform -t ya-agent-platform:dev .

.PHONY: docker-run-platform
docker-run-platform: ## Run the YA Agent Platform Docker image
	@echo "Running ya-agent-platform Docker image"
	@docker run --rm ya-agent-platform:dev

.PHONY: check
check: ## Run code quality tools for all active packages
	@echo "Checking lock file consistency with pyproject.toml"
	@uv lock --locked
	@echo "Running pre-commit"
	@SKIP=cargo-fmt-desktop,cargo-clippy-desktop uv run pre-commit run -a
	@echo "Checking bundled skills sync"
	@./scripts/check-skills-sync.sh
	@echo "Checking release skill zip build"
	@python scripts/build-skill-zips.py --check
	@echo "Running web lint"
	@$(MAKE) web-lint
	@echo "Running web build"
	@$(MAKE) web-build
	@echo "Running desktop lint"
	@$(MAKE) desktop-lint
	@echo "Running desktop build"
	@$(MAKE) desktop-build
	@echo "Checking desktop Rust formatting"
	@$(MAKE) desktop-rust-fmt
	@echo "Running desktop Rust check"
	@$(MAKE) desktop-rust-check
	@echo "Running desktop Rust clippy"
	@$(MAKE) desktop-rust-clippy
	@echo "Running pyright"
	@uv run python -m pyright
	@echo "Running deptry for ya-agent-environment"
	@(cd packages/ya-agent-environment && uvx deptry ya_agent_environment)
	@echo "Running deptry for ya-environment-relay"
	@(cd packages/ya-environment-relay && uvx deptry ya_environment_relay)
	@echo "Running deptry for ya-agent-sdk"
	@(cd packages/ya-agent-sdk && uvx deptry ya_agent_sdk)
	@echo "Running deptry for yaacli"
	@(cd packages/yaacli && uvx deptry yaacli)
	@echo "Running deptry for ya-claw"
	@(cd packages/ya-claw && uvx deptry ya_claw)

.PHONY: test
test: ## Run environment, SDK, CLI, YA Claw, and YA Desktop tests
	@echo "Running pytest for workspace packages"
	@uv run python -m pytest packages/ya-agent-environment/tests packages/ya-agent-sdk/tests packages/yaacli/tests packages/ya-claw/tests -n auto -vv --inline-snapshot=disable --cov --cov-config=pyproject.toml --cov-report term-missing
	@echo "Running YA Desktop tests"
	@$(MAKE) desktop-test

.PHONY: test-environment
test-environment: ## Run ya-agent-environment tests
	@echo "Running ya-agent-environment pytest"
	@uv run python -m pytest packages/ya-agent-environment/tests -n auto -vv --cov --cov-config=pyproject.toml --cov-report term-missing

.PHONY: test-sdk
test-sdk: ## Run SDK tests
	@echo "Running SDK pytest"
	@uv run python -m pytest packages/ya-agent-sdk/tests -n auto -vv --inline-snapshot=disable --cov --cov-config=pyproject.toml --cov-report term-missing

.PHONY: test-cli
test-cli: ## Run CLI tests
	@echo "Running CLI pytest"
	@uv run python -m pytest packages/yaacli/tests -n auto -vv --inline-snapshot=disable

.PHONY: test-claw
test-claw: ## Run YA Claw tests
	@echo "Running YA Claw pytest"
	@uv run python -m pytest packages/ya-claw/tests -n auto -vv --inline-snapshot=disable --cov --cov-config=pyproject.toml --cov-report term-missing

.PHONY: claw-smoke
claw-smoke: ## Run YA Claw HTTP smoke test against the configured local server
	@echo "Running YA Claw smoke test"
	@sh packages/ya-claw/scripts/e2e_smoke.sh

.PHONY: claw-sse-close-smoke
claw-sse-close-smoke: ## Run YA Claw SSE cancel/close smoke test against the configured local server
	@echo "Running YA Claw SSE close smoke test"
	@sh packages/ya-claw/scripts/e2e_sse_close.sh

.PHONY: claw-sse-complete-smoke
claw-sse-complete-smoke: ## Run YA Claw SSE completion smoke test against the configured local server
	@echo "Running YA Claw SSE completion smoke test"
	@sh packages/ya-claw/scripts/e2e_sse_complete.sh

.PHONY: test-fix
test-fix: ## Run pytest with inline snapshot updates
	@echo "Running pytest with inline snapshot updates"
	@uv run python -m pytest packages/ya-agent-environment/tests packages/ya-agent-sdk/tests packages/yaacli/tests packages/ya-claw/tests -vv --inline-snapshot=fix

.PHONY: build
build: clean-build ## Build ya-agent-sdk distribution
	@echo "Building ya-agent-sdk"
	@uv build --package ya-agent-sdk -o dist

.PHONY: build-environment
build-environment: clean-build ## Build ya-agent-environment distribution
	@echo "Building ya-agent-environment"
	@uv build --package ya-agent-environment -o dist

.PHONY: build-relay
build-relay: clean-build ## Build ya-environment-relay distribution
	@echo "Building ya-environment-relay"
	@uv build --package ya-environment-relay -o dist

.PHONY: build-claw
build-claw: clean-build ## Build ya-claw distribution
	@echo "Building ya-claw"
	@uv build --package ya-claw -o dist

.PHONY: build-platform
build-platform: clean-build ## Build the ya-agent-platform package
	@echo "Building ya-agent-platform package"
	@uv build --package ya-agent-platform -o dist

.PHONY: build-all
build-all: clean-build ## Build distributions for all workspace packages
	@echo "Building workspace packages"
	@uv build --all-packages -o dist

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "Removing build artifacts"
	@uv run python -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in (Path('dist'), Path('packages/ya-agent-environment/dist'), Path('packages/ya-environment-relay/dist'), Path('packages/ya-agent-sdk/dist'), Path('packages/yaacli/dist'), Path('packages/ya-claw/dist'), Path('packages/ya-agent-platform/dist'))]"

.PHONY: publish
publish: ## Publish built distributions to PyPI
	@echo "Publishing distributions"
	@uv publish dist/*

.PHONY: build-and-publish
build-and-publish: build-all publish ## Build and publish all workspace packages.

.PHONY: help
help:
	@uv run python -c "import re; [[print(f'\033[36m{m[0]:<24}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
