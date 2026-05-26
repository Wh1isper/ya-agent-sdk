use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, RunEvent, State};

const LOCAL_CLAW_READY_TIMEOUT: Duration = Duration::from_secs(30);
const CLAW_AUTO_UPDATE_INITIAL_DELAY: Duration = Duration::from_secs(30);
const CLAW_AUTO_UPDATE_INTERVAL_SECONDS: u64 = 24 * 60 * 60;
const DESKTOP_CLAW_CONTRACT: &str = "claw-desktop.v1";
const DESKTOP_RELAY_PROTOCOL: &str = "ya-environment-relay.v1";
const DEFAULT_CLAW_PACKAGE_SPEC: &str = "ya-claw[rs]";
const FALLBACK_CLAW_PACKAGE_SPEC: &str = "ya-claw";
const DEFAULT_CLAW_PYTHON_VERSION: &str = "3.13";

#[derive(Default)]
struct LocalClawManager {
    process: Mutex<Option<LocalClawProcess>>,
}

#[derive(Default)]
struct RuntimeManager {
    lock: Mutex<()>,
}

struct LocalClawProcess {
    child: Child,
    info: LocalClawRuntimeInfo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocalClawRuntimeInfo {
    base_url: String,
    pid: u32,
    data_dir: String,
    workspace_dir: String,
    sqlite_path: String,
    log_file: String,
    lock_file: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalClawStatus {
    running: bool,
    base_url: Option<String>,
    pid: Option<u32>,
    data_dir: Option<String>,
    workspace_dir: Option<String>,
    sqlite_path: Option<String>,
    log_file: Option<String>,
    lock_file: Option<String>,
    api_token: Option<String>,
    profile_seed_file: Option<String>,
    relay_protocol: String,
    message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocalClawEnvVar {
    key: String,
    value: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LocalClawLaunchConfig {
    #[serde(default = "default_true")]
    agency_enabled: bool,
    #[serde(default = "default_true")]
    memory_enabled: bool,
    #[serde(default = "default_true")]
    shell_review_enabled: bool,
    #[serde(default = "default_shell_review_model")]
    shell_review_model: String,
    #[serde(default = "default_shell_review_model_settings")]
    shell_review_model_settings: String,
    #[serde(default = "default_shell_review_risk_threshold")]
    shell_review_risk_threshold: String,
    #[serde(default = "default_shell_review_unattended_risk_threshold")]
    shell_review_unattended_risk_threshold: String,
    #[serde(default = "default_shell_review_action")]
    shell_review_action: String,
    #[serde(default = "default_true")]
    shell_sandbox_enabled: bool,
    #[serde(default = "default_shell_sandbox_backend")]
    shell_sandbox_backend: String,
    #[serde(default = "default_shell_sandbox_network")]
    shell_sandbox_network: String,
    #[serde(default)]
    shell_sandbox_allow_raw_host: bool,
    #[serde(default)]
    preset_name: Option<String>,
    #[serde(default)]
    env: Vec<LocalClawEnvVar>,
    #[serde(default)]
    config_file: Option<String>,
}

impl Default for LocalClawLaunchConfig {
    fn default() -> Self {
        Self {
            agency_enabled: true,
            memory_enabled: true,
            shell_review_enabled: true,
            shell_review_model: default_shell_review_model(),
            shell_review_model_settings: default_shell_review_model_settings(),
            shell_review_risk_threshold: default_shell_review_risk_threshold(),
            shell_review_unattended_risk_threshold: default_shell_review_unattended_risk_threshold(
            ),
            shell_review_action: default_shell_review_action(),
            shell_sandbox_enabled: true,
            shell_sandbox_backend: default_shell_sandbox_backend(),
            shell_sandbox_network: default_shell_sandbox_network(),
            shell_sandbox_allow_raw_host: false,
            preset_name: Some("Desktop default".to_string()),
            env: Vec::new(),
            config_file: None,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_shell_review_model() -> String {
    "gateway@openai-responses:gpt-5.4-mini".to_string()
}

fn default_shell_review_model_settings() -> String {
    "openai_responses_low".to_string()
}

fn default_shell_review_risk_threshold() -> String {
    "extra_high".to_string()
}

fn default_shell_review_unattended_risk_threshold() -> String {
    "extra_high".to_string()
}

fn default_shell_review_action() -> String {
    "defer".to_string()
}

fn default_shell_sandbox_backend() -> String {
    "auto".to_string()
}

fn default_shell_sandbox_network() -> String {
    "full".to_string()
}

#[derive(Debug, Deserialize)]
struct ReadyLine {
    #[serde(rename = "type")]
    event_type: String,
    pid: u32,
    base_url: String,
    data_dir: Option<String>,
    workspace_dir: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActiveClawRuntime {
    entrypoint: String,
    #[serde(default)]
    runtime_dir: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    package_spec: String,
    #[serde(default)]
    python_version: String,
    #[serde(default)]
    uv_path: String,
    #[serde(default)]
    installed_at: u64,
    #[serde(default)]
    contract: String,
}

#[derive(Debug)]
struct RuntimeManagerLayout {
    claw_dir: PathBuf,
    logs_dir: PathBuf,
    active_file: PathBuf,
    previous_active_file: PathBuf,
    update_state_file: PathBuf,
    uv_cache_dir: PathBuf,
    uv_python_dir: PathBuf,
    app_uv_dir: PathBuf,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeManagerStatus {
    active: Option<ActiveClawRuntime>,
    runtimes: Vec<InstalledClawRuntime>,
    uv_path: Option<String>,
    claw_dir: String,
    logs_dir: String,
    update_state: RuntimeUpdateState,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InstalledClawRuntime {
    id: String,
    runtime_dir: String,
    version: Option<String>,
    active: bool,
    failed: bool,
    log_file: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeInstallResult {
    runtime: ActiveClawRuntime,
    log_file: String,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeActionResult {
    success: bool,
    message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeUpdateState {
    last_checked_at: Option<u64>,
    next_check_after: Option<u64>,
    check_in_progress: bool,
    update_ready: bool,
    candidate: Option<ActiveClawRuntime>,
    last_error: Option<String>,
    last_log_file: Option<String>,
    auto_update_enabled: bool,
}

impl Default for RuntimeUpdateState {
    fn default() -> Self {
        Self {
            last_checked_at: None,
            next_check_after: None,
            check_in_progress: false,
            update_ready: false,
            candidate: None,
            last_error: None,
            last_log_file: None,
            auto_update_enabled: true,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeUpdateCheckResult {
    update_ready: bool,
    candidate: Option<ActiveClawRuntime>,
    log_file: Option<String>,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopWorkspaceStatus {
    workspace_root: String,
    profile_seed_file: String,
    relay_protocol: String,
    shell_review_enabled: bool,
    shell_review_risk_threshold: String,
    shell_review_unattended_risk_threshold: String,
    shell_review_action: String,
    shell_sandbox_enabled: bool,
    shell_sandbox_backend: String,
    shell_sandbox_network: String,
    shell_sandbox_allow_raw_host: bool,
    message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopOnboardingResult {
    config: LocalClawLaunchConfig,
    workspace_status: DesktopWorkspaceStatus,
    api_token_configured: bool,
    message: String,
}

#[derive(Debug, Deserialize)]
struct ClawdVersionPayload {
    version: String,
    desktop_compatibility: DesktopCompatibility,
}

#[derive(Debug, Deserialize)]
struct DesktopCompatibility {
    contract: String,
}

enum ProcessLine {
    Stdout(String),
    Stderr(String),
}

#[tauri::command]
fn get_local_claw_status(state: State<'_, LocalClawManager>) -> Result<LocalClawStatus, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "Local Claw process lock is poisoned".to_string())?;
    let Some(process) = guard.as_mut() else {
        return Ok(stopped_status("Local Claw sidecar is stopped"));
    };

    match process.child.try_wait() {
        Ok(Some(status)) => {
            let code = status
                .code()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "signal".to_string());
            let info = process.info.clone();
            *guard = None;
            Ok(status_from_info(
                &info,
                false,
                &format!("Local Claw sidecar exited with {}", code),
            ))
        }
        Ok(None) => Ok(status_from_info(
            &process.info,
            true,
            "Local Claw sidecar is running",
        )),
        Err(error) => Ok(status_from_info(
            &process.info,
            true,
            &format!("Local Claw status check failed: {}", error),
        )),
    }
}

#[tauri::command]
fn start_local_claw(
    app: AppHandle,
    state: State<'_, LocalClawManager>,
) -> Result<LocalClawStatus, String> {
    {
        let mut guard = state
            .process
            .lock()
            .map_err(|_| "Local Claw process lock is poisoned".to_string())?;
        if let Some(process) = guard.as_mut() {
            if process
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_none()
            {
                return Ok(status_from_info(
                    &process.info,
                    true,
                    "Local Claw sidecar is already running",
                ));
            }
            *guard = None;
        }
    }

    let layout = ensure_local_claw_layout(&app)?;
    let api_token = ensure_local_api_token(&layout.env_file)?;
    let launch_config = read_local_claw_launch_config(&layout.launch_config_file)?;
    let profile_seed_file = ensure_desktop_profile_seed_file(&layout, &launch_config)?;
    let local_port = reserve_local_claw_port()?;
    let public_base_url = format!("http://127.0.0.1:{}", local_port);
    let command_spec = resolve_clawd_command(&app)?;
    let mut command = Command::new(&command_spec.program);
    command.args(&command_spec.args);
    command.args([
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        &local_port.to_string(),
        "--data-dir",
        layout.data_dir.to_string_lossy().as_ref(),
        "--sqlite-path",
        layout.sqlite_path.to_string_lossy().as_ref(),
        "--workspace-root",
        layout.workspace_dir.to_string_lossy().as_ref(),
        "--runtime-lock-file",
        layout.lock_file.to_string_lossy().as_ref(),
        "--ready-json",
    ]);
    command.env("YA_CLAW_ENVIRONMENT", "desktop-local");
    command.env(
        "YA_CLAW_ALLOW_ORIGINS",
        r#"["http://127.0.0.1:5174","http://localhost:5174","http://tauri.localhost","https://tauri.localhost"]"#,
    );
    command.env("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual");
    command.env("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local");
    command.env("YA_CLAW_AUTO_SEED_PROFILES", "true");
    command.env(
        "YA_CLAW_AGENCY_ENABLED",
        bool_env(launch_config.agency_enabled),
    );
    command.env(
        "YA_CLAW_MEMORY_ENABLED",
        bool_env(launch_config.memory_enabled),
    );
    command.env(
        "YA_CLAW_SHELL_SANDBOX_ENABLED",
        bool_env(launch_config.shell_sandbox_enabled),
    );
    command.env(
        "YA_CLAW_SHELL_SANDBOX_BACKEND",
        &launch_config.shell_sandbox_backend,
    );
    command.env(
        "YA_CLAW_SHELL_SANDBOX_NETWORK",
        &launch_config.shell_sandbox_network,
    );
    command.env(
        "YA_CLAW_SHELL_SANDBOX_ALLOW_RAW_HOST",
        bool_env(launch_config.shell_sandbox_allow_raw_host),
    );
    for entry in &launch_config.env {
        command.env(&entry.key, &entry.value);
    }
    command.env("YA_CLAW_PUBLIC_BASE_URL", &public_base_url);
    command.env("YA_CLAW_API_TOKEN", api_token);
    command.env("YA_CLAW_PROFILE_SEED_FILE", profile_seed_file);
    if let Some(seed_file) = resolve_profile_seed_file(&app) {
        command.env("YA_CLAW_PROFILE_SEED_FILE", seed_file);
    }
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|error| {
        format!(
            "Failed to start Local Claw sidecar with {}: {}",
            command_spec.display, error
        )
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture Local Claw stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture Local Claw stderr".to_string())?;

    let child_pid = child.id();
    let ready_info = wait_for_ready(
        &mut child,
        stdout,
        stderr,
        LocalClawRuntimeInfo {
            base_url: String::new(),
            pid: child_pid,
            data_dir: layout.data_dir.to_string_lossy().to_string(),
            workspace_dir: layout.workspace_dir.to_string_lossy().to_string(),
            sqlite_path: layout.sqlite_path.to_string_lossy().to_string(),
            log_file: layout.log_file.to_string_lossy().to_string(),
            lock_file: layout.lock_file.to_string_lossy().to_string(),
        },
        layout.log_file.clone(),
    )?;

    let mut guard = state
        .process
        .lock()
        .map_err(|_| "Local Claw process lock is poisoned".to_string())?;
    *guard = Some(LocalClawProcess {
        child,
        info: ready_info.clone(),
    });

    Ok(status_from_info(
        &ready_info,
        true,
        "Local Claw sidecar started",
    ))
}

#[tauri::command]
fn stop_local_claw(state: State<'_, LocalClawManager>) -> Result<LocalClawStatus, String> {
    stop_local_claw_with_message(state, "Local Claw sidecar stopped")
}

fn stop_local_claw_with_message(
    state: State<'_, LocalClawManager>,
    message: &str,
) -> Result<LocalClawStatus, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "Local Claw process lock is poisoned".to_string())?;
    let Some(mut process) = guard.take() else {
        return Ok(stopped_status("Local Claw sidecar is stopped"));
    };

    let info = process.info.clone();
    if process
        .child
        .try_wait()
        .map_err(|error| error.to_string())?
        .is_none()
    {
        process.child.kill().map_err(|error| error.to_string())?;
        let _ = process.child.wait();
    }
    Ok(status_from_info(&info, false, message))
}

#[tauri::command]
fn restart_local_claw(
    app: AppHandle,
    state: State<'_, LocalClawManager>,
) -> Result<LocalClawStatus, String> {
    let _ = stop_local_claw_with_message(state.clone(), "Local Claw sidecar restarting");
    start_local_claw(app, state)
}

#[tauri::command]
fn get_runtime_manager_status(app: AppHandle) -> Result<RuntimeManagerStatus, String> {
    let layout = ensure_runtime_manager_layout(&app)?;
    let active = read_active_runtime(&layout.active_file).ok();
    let uv_path = resolve_uv_path(&app, &layout).ok();
    let runtimes = list_installed_claw_runtimes(&layout, active.as_ref())?;

    let update_state = read_runtime_update_state(&layout)?;

    Ok(RuntimeManagerStatus {
        active,
        runtimes,
        uv_path: uv_path.map(|path| path.to_string_lossy().to_string()),
        claw_dir: layout.claw_dir.to_string_lossy().to_string(),
        logs_dir: layout.logs_dir.to_string_lossy().to_string(),
        update_state,
        message: "Runtime Manager status loaded".to_string(),
    })
}

#[tauri::command]
fn install_latest_claw_runtime(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    local_claw: State<'_, LocalClawManager>,
) -> Result<RuntimeInstallResult, String> {
    install_or_update_claw_runtime(app, runtime_manager, local_claw, false)
}

#[tauri::command]
fn update_claw_runtime(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    local_claw: State<'_, LocalClawManager>,
) -> Result<RuntimeInstallResult, String> {
    install_or_update_claw_runtime(app, runtime_manager, local_claw, true)
}

#[tauri::command]
fn repair_claw_runtime(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    local_claw: State<'_, LocalClawManager>,
    version: Option<String>,
) -> Result<RuntimeInstallResult, String> {
    if let Some(value) = version
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let layout = ensure_runtime_manager_layout(&app)?;
        if let Some(runtime_dir) = find_claw_runtime_dir(&layout, value)? {
            let _ = fs::write(
                runtime_dir.join("repair-requested.txt"),
                "Repair requested by YA Desktop\n",
            );
        }
    }
    install_or_update_claw_runtime(app, runtime_manager, local_claw, true)
}

#[tauri::command]
fn remove_claw_runtime(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    version: String,
) -> Result<RuntimeActionResult, String> {
    let _guard = runtime_manager
        .lock
        .lock()
        .map_err(|_| "Runtime Manager lock is poisoned".to_string())?;
    let layout = ensure_runtime_manager_layout(&app)?;
    let runtime_dir = find_claw_runtime_dir(&layout, &version)?
        .ok_or_else(|| format!("Claw runtime {} was not found", version))?;
    let active = read_active_runtime(&layout.active_file).ok();
    if active
        .as_ref()
        .map(|metadata| Path::new(&metadata.runtime_dir) == runtime_dir)
        .unwrap_or(false)
    {
        return Err(
            "Active Claw runtime can be removed after another runtime is activated".to_string(),
        );
    }
    fs::remove_dir_all(&runtime_dir).map_err(|error| error.to_string())?;
    Ok(RuntimeActionResult {
        success: true,
        message: format!("Removed Claw runtime {}", version),
    })
}

#[tauri::command]
fn check_claw_runtime_update(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
) -> Result<RuntimeUpdateCheckResult, String> {
    check_claw_runtime_update_inner(app, runtime_manager, true)
}

#[tauri::command]
fn apply_ready_claw_runtime_update(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    local_claw: State<'_, LocalClawManager>,
) -> Result<LocalClawStatus, String> {
    let _guard = runtime_manager
        .lock
        .lock()
        .map_err(|_| "Runtime Manager lock is poisoned".to_string())?;
    let layout = ensure_runtime_manager_layout(&app)?;
    let mut update_state = read_runtime_update_state(&layout)?;
    let candidate = update_state
        .candidate
        .clone()
        .filter(|runtime| update_state.update_ready && Path::new(&runtime.entrypoint).exists())
        .ok_or_else(|| "No verified Claw runtime update is ready to apply".to_string())?;

    let _ = stop_local_claw_with_message(
        local_claw.clone(),
        "Local Claw sidecar stopped for runtime update",
    );
    activate_claw_runtime(&layout, &candidate)?;
    update_state.update_ready = false;
    update_state.candidate = None;
    update_state.last_error = None;
    write_runtime_update_state(&layout, &update_state)?;
    start_local_claw(app, local_claw)
}

#[tauri::command]
fn get_local_claw_launch_config(app: AppHandle) -> Result<LocalClawLaunchConfig, String> {
    let layout = ensure_local_claw_layout(&app)?;
    read_local_claw_launch_config(&layout.launch_config_file)
}

#[tauri::command]
fn update_local_claw_launch_config(
    app: AppHandle,
    config: LocalClawLaunchConfig,
) -> Result<LocalClawLaunchConfig, String> {
    let layout = ensure_local_claw_layout(&app)?;
    let config = normalize_launch_config(config)?;
    write_local_claw_launch_config(&layout.launch_config_file, &config)?;
    let _ = ensure_desktop_profile_seed_file(&layout, &config)?;
    read_local_claw_launch_config(&layout.launch_config_file)
}

#[tauri::command]
fn reset_local_claw_launch_config(app: AppHandle) -> Result<LocalClawLaunchConfig, String> {
    let layout = ensure_local_claw_layout(&app)?;
    let config = with_config_file(LocalClawLaunchConfig::default(), &layout.launch_config_file);
    write_local_claw_launch_config(&layout.launch_config_file, &config)?;
    Ok(config)
}

#[tauri::command]
fn import_local_claw_launch_preset(
    app: AppHandle,
    raw: String,
) -> Result<LocalClawLaunchConfig, String> {
    let layout = ensure_local_claw_layout(&app)?;
    let config = parse_launch_preset(&raw)?;
    write_local_claw_launch_config(&layout.launch_config_file, &config)?;
    let _ = ensure_desktop_profile_seed_file(&layout, &config)?;
    read_local_claw_launch_config(&layout.launch_config_file)
}

#[tauri::command]
fn get_desktop_workspace_status(app: AppHandle) -> Result<DesktopWorkspaceStatus, String> {
    let layout = ensure_local_claw_layout(&app)?;
    let config = read_local_claw_launch_config(&layout.launch_config_file)?;
    desktop_workspace_status_from_layout(&layout, &config)
}

#[tauri::command]
fn run_desktop_onboarding(
    app: AppHandle,
    config: Option<LocalClawLaunchConfig>,
) -> Result<DesktopOnboardingResult, String> {
    let layout = ensure_local_claw_layout(&app)?;
    let config = normalize_launch_config(config.unwrap_or_default())?;
    write_local_claw_launch_config(&layout.launch_config_file, &config)?;
    ensure_local_api_token(&layout.env_file)?;
    let workspace_status = desktop_workspace_status_from_layout(&layout, &config)?;
    let config = read_local_claw_launch_config(&layout.launch_config_file)?;
    Ok(DesktopOnboardingResult {
        config,
        workspace_status,
        api_token_configured: true,
        message: "Desktop onboarding initialized Local Claw configuration".to_string(),
    })
}

#[tauri::command]
fn get_runtime_install_log(app: AppHandle, version: Option<String>) -> Result<String, String> {
    let layout = ensure_runtime_manager_layout(&app)?;
    if let Some(value) = version
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let log_file = layout
            .logs_dir
            .join(format!("{}.log", sanitize_runtime_id(value)));
        if log_file.exists() {
            return fs::read_to_string(log_file).map_err(|error| error.to_string());
        }
        if let Some(runtime_dir) = find_claw_runtime_dir(&layout, value)? {
            let runtime_log = runtime_dir.join("install.log");
            if runtime_log.exists() {
                return fs::read_to_string(runtime_log).map_err(|error| error.to_string());
            }
        }
        return Err(format!("Install log for {} was not found", value));
    }

    let mut logs = fs::read_dir(&layout.logs_dir)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("log"))
        .collect::<Vec<_>>();
    logs.sort_by_key(|entry| {
        entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    let Some(entry) = logs.pop() else {
        return Ok(String::new());
    };
    fs::read_to_string(entry.path()).map_err(|error| error.to_string())
}

fn check_claw_runtime_update_inner(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    force: bool,
) -> Result<RuntimeUpdateCheckResult, String> {
    let _guard = runtime_manager
        .lock
        .lock()
        .map_err(|_| "Runtime Manager lock is poisoned".to_string())?;
    let layout = ensure_runtime_manager_layout(&app)?;
    let active = read_active_runtime(&layout.active_file).ok();
    let now = unix_timestamp()?;
    let mut update_state = read_runtime_update_state(&layout)?;

    if !update_state.auto_update_enabled && !force {
        return Ok(RuntimeUpdateCheckResult {
            update_ready: update_state.update_ready,
            candidate: update_state.candidate,
            log_file: update_state.last_log_file,
            message: "Claw runtime auto-update is disabled".to_string(),
        });
    }

    if !force
        && update_state.update_ready
        && update_state
            .candidate
            .as_ref()
            .map(|runtime| Path::new(&runtime.entrypoint).exists())
            .unwrap_or(false)
    {
        return Ok(RuntimeUpdateCheckResult {
            update_ready: true,
            candidate: update_state.candidate,
            log_file: update_state.last_log_file,
            message: "Verified Claw runtime update is ready to apply".to_string(),
        });
    }

    if !force
        && update_state
            .next_check_after
            .map(|next_check| now < next_check)
            .unwrap_or(false)
    {
        return Ok(RuntimeUpdateCheckResult {
            update_ready: update_state.update_ready,
            candidate: update_state.candidate,
            log_file: update_state.last_log_file,
            message: "Claw runtime update check is scheduled for later".to_string(),
        });
    }

    update_state.check_in_progress = true;
    update_state.last_checked_at = Some(now);
    update_state.next_check_after = Some(now + CLAW_AUTO_UPDATE_INTERVAL_SECONDS);
    update_state.last_error = None;
    write_runtime_update_state(&layout, &update_state)?;

    if let Some(candidate) = update_state.candidate.as_ref() {
        let candidate_dir = Path::new(&candidate.runtime_dir);
        let active_dir = active.as_ref().map(|runtime| runtime.runtime_dir.as_str());
        if candidate_dir.exists() && active_dir != Some(candidate.runtime_dir.as_str()) {
            let _ = fs::remove_dir_all(candidate_dir);
        }
    }

    let runtime_id = format!("candidate-{}", now);
    let runtime_dir = layout.claw_dir.join(&runtime_id);
    let log_file = layout.logs_dir.join(format!("{}.log", runtime_id));
    fs::create_dir_all(&runtime_dir).map_err(|error| error.to_string())?;
    append_log_line(
        &log_file,
        "Starting Claw runtime update candidate installation",
    );

    let result = match install_claw_runtime_inner(&app, &layout, &runtime_dir, &log_file, true) {
        Ok(candidate) => {
            let same_version = active
                .as_ref()
                .map(|runtime| runtime.version == candidate.version)
                .unwrap_or(false);
            if same_version {
                append_log_line(
                    &log_file,
                    &format!("Active Claw runtime is already at {}", candidate.version),
                );
                let _ = fs::remove_dir_all(&runtime_dir);
                update_state.check_in_progress = false;
                update_state.update_ready = false;
                update_state.candidate = None;
                update_state.last_log_file = Some(log_file.to_string_lossy().to_string());
                write_runtime_update_state(&layout, &update_state)?;
                RuntimeUpdateCheckResult {
                    update_ready: false,
                    candidate: None,
                    log_file: Some(log_file.to_string_lossy().to_string()),
                    message: "Active Claw runtime is already latest".to_string(),
                }
            } else {
                update_state.check_in_progress = false;
                update_state.update_ready = true;
                update_state.candidate = Some(candidate.clone());
                update_state.last_log_file = Some(log_file.to_string_lossy().to_string());
                update_state.last_error = None;
                write_runtime_update_state(&layout, &update_state)?;
                RuntimeUpdateCheckResult {
                    update_ready: true,
                    candidate: Some(candidate),
                    log_file: Some(log_file.to_string_lossy().to_string()),
                    message: "Claw runtime update is ready to apply".to_string(),
                }
            }
        }
        Err(error) => {
            let _ = fs::write(
                runtime_dir.join("failed.json"),
                serde_json::json!({ "error": &error }).to_string(),
            );
            append_log_line(&log_file, &format!("Update check failed: {}", error));
            update_state.check_in_progress = false;
            update_state.last_error = Some(error.clone());
            update_state.last_log_file = Some(log_file.to_string_lossy().to_string());
            write_runtime_update_state(&layout, &update_state)?;
            RuntimeUpdateCheckResult {
                update_ready: update_state.update_ready,
                candidate: update_state.candidate,
                log_file: Some(log_file.to_string_lossy().to_string()),
                message: format!("Claw runtime update check failed: {}", error),
            }
        }
    };

    Ok(result)
}

fn install_or_update_claw_runtime(
    app: AppHandle,
    runtime_manager: State<'_, RuntimeManager>,
    local_claw: State<'_, LocalClawManager>,
    upgrade: bool,
) -> Result<RuntimeInstallResult, String> {
    let _guard = runtime_manager
        .lock
        .lock()
        .map_err(|_| "Runtime Manager lock is poisoned".to_string())?;
    let layout = ensure_runtime_manager_layout(&app)?;
    let runtime_id = format!("runtime-{}", unix_timestamp()?);
    let runtime_dir = layout.claw_dir.join(&runtime_id);
    let log_file = layout.logs_dir.join(format!("{}.log", runtime_id));
    fs::create_dir_all(&runtime_dir).map_err(|error| error.to_string())?;
    append_log_line(&log_file, "Starting Claw runtime installation");

    match install_claw_runtime_inner(&app, &layout, &runtime_dir, &log_file, upgrade) {
        Ok(runtime) => {
            let _ = stop_local_claw_with_message(
                local_claw,
                "Local Claw sidecar stopped for runtime activation",
            );
            activate_claw_runtime(&layout, &runtime)?;
            let mut update_state = read_runtime_update_state(&layout)?;
            update_state.update_ready = false;
            update_state.candidate = None;
            update_state.last_error = None;
            write_runtime_update_state(&layout, &update_state)?;
            Ok(RuntimeInstallResult {
                runtime,
                log_file: log_file.to_string_lossy().to_string(),
                message: "Claw runtime installed and activated".to_string(),
            })
        }
        Err(error) => {
            let _ = fs::write(
                runtime_dir.join("failed.json"),
                serde_json::json!({ "error": &error }).to_string(),
            );
            append_log_line(&log_file, &format!("Install failed: {}", error));
            Err(format!(
                "Claw runtime installation failed; see {}: {}",
                log_file.display(),
                error
            ))
        }
    }
}

fn install_claw_runtime_inner(
    app: &AppHandle,
    layout: &RuntimeManagerLayout,
    runtime_dir: &Path,
    log_file: &Path,
    upgrade: bool,
) -> Result<ActiveClawRuntime, String> {
    let uv_path = resolve_uv_path(app, layout)?;
    let package_spec = std::env::var("YA_DESKTOP_CLAW_PACKAGE_SPEC")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_CLAW_PACKAGE_SPEC.to_string());
    let fallback_package_spec = std::env::var("YA_DESKTOP_CLAW_FALLBACK_PACKAGE_SPEC")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| FALLBACK_CLAW_PACKAGE_SPEC.to_string());
    let allow_rs_fallback =
        package_spec == DEFAULT_CLAW_PACKAGE_SPEC && fallback_package_spec != package_spec;
    let python_version = std::env::var("YA_DESKTOP_CLAW_PYTHON_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_CLAW_PYTHON_VERSION.to_string());
    let venv_dir = runtime_dir.join(".venv");
    let python_path = runtime_python_path(&venv_dir);
    let entrypoint = runtime_clawd_path(&venv_dir);

    run_uv_command(
        &uv_path,
        layout,
        [
            "python".to_string(),
            "install".to_string(),
            python_version.clone(),
        ]
        .as_slice(),
        log_file,
    )?;
    run_uv_command(
        &uv_path,
        layout,
        [
            "venv".to_string(),
            venv_dir.to_string_lossy().to_string(),
            "--python".to_string(),
            python_version.clone(),
        ]
        .as_slice(),
        log_file,
    )?;

    let mut pip_args = vec![
        "pip".to_string(),
        "install".to_string(),
        "--python".to_string(),
        python_path.to_string_lossy().to_string(),
    ];
    if upgrade {
        pip_args.push("--upgrade".to_string());
    }
    let installed_package_spec = install_runtime_package(
        &uv_path,
        layout,
        &pip_args,
        &package_spec,
        if allow_rs_fallback {
            Some(fallback_package_spec.as_str())
        } else {
            None
        },
        log_file,
    )?;

    let version_output = run_runtime_command_capture(
        &entrypoint,
        ["version", "--json-output"].as_slice(),
        log_file,
    )?;
    let version_payload: ClawdVersionPayload = serde_json::from_str(version_output.trim())
        .map_err(|error| format!("Failed to parse ya-clawd version output: {}", error))?;
    if version_payload.desktop_compatibility.contract != DESKTOP_CLAW_CONTRACT {
        return Err(format!(
            "Claw runtime contract {} does not match supported contract {}",
            version_payload.desktop_compatibility.contract, DESKTOP_CLAW_CONTRACT
        ));
    }

    let verify_token = format!("desktop-verify-{}", unix_timestamp()?);
    run_runtime_command_with_env(
        &entrypoint,
        ["doctor", "--json-output"].as_slice(),
        &[
            ("YA_CLAW_API_TOKEN", verify_token.as_str()),
            ("YA_CLAW_ENVIRONMENT", "desktop-local"),
            ("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual"),
            ("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local"),
        ],
        log_file,
    )?;

    let runtime = ActiveClawRuntime {
        entrypoint: entrypoint.to_string_lossy().to_string(),
        runtime_dir: runtime_dir.to_string_lossy().to_string(),
        version: version_payload.version,
        package_spec: installed_package_spec,
        python_version,
        uv_path: uv_path.to_string_lossy().to_string(),
        installed_at: unix_timestamp()?,
        contract: DESKTOP_CLAW_CONTRACT.to_string(),
    };
    write_json_atomic(&runtime_dir.join("runtime.json"), &runtime)?;
    fs::write(
        runtime_dir.join("install.log"),
        fs::read_to_string(log_file).unwrap_or_default(),
    )
    .map_err(|error| error.to_string())?;
    Ok(runtime)
}

fn install_runtime_package(
    uv_path: &Path,
    layout: &RuntimeManagerLayout,
    base_pip_args: &[String],
    package_spec: &str,
    fallback_package_spec: Option<&str>,
    log_file: &Path,
) -> Result<String, String> {
    let mut primary_args = base_pip_args.to_vec();
    primary_args.push(package_spec.to_string());
    match run_uv_command(uv_path, layout, &primary_args, log_file) {
        Ok(()) => Ok(package_spec.to_string()),
        Err(primary_error) => {
            let Some(fallback) = fallback_package_spec else {
                return Err(primary_error);
            };
            append_log_line(
                log_file,
                &format!(
                    "Rust runtime install failed for {}; retrying with {}: {}",
                    package_spec, fallback, primary_error
                ),
            );
            let mut fallback_args = base_pip_args.to_vec();
            fallback_args.push(fallback.to_string());
            run_uv_command(uv_path, layout, &fallback_args, log_file)?;
            Ok(fallback.to_string())
        }
    }
}

fn ensure_runtime_manager_layout(app: &AppHandle) -> Result<RuntimeManagerLayout, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve app data dir: {}", error))?;
    let root_dir = app_data_dir.join("runtimes");
    let claw_dir = root_dir.join("claw");
    let logs_dir = claw_dir.join("logs");
    let uv_cache_dir = app_data_dir.join("uv").join("cache");
    let uv_python_dir = app_data_dir.join("uv").join("python");
    let app_uv_dir = app_data_dir.join("uv").join("bin");
    fs::create_dir_all(&claw_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&logs_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&uv_cache_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&uv_python_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&app_uv_dir).map_err(|error| error.to_string())?;
    Ok(RuntimeManagerLayout {
        active_file: claw_dir.join("active.json"),
        previous_active_file: claw_dir.join("previous-active.json"),
        update_state_file: claw_dir.join("update-state.json"),
        claw_dir,
        logs_dir,
        uv_cache_dir,
        uv_python_dir,
        app_uv_dir,
    })
}

fn resolve_uv_path(app: &AppHandle, layout: &RuntimeManagerLayout) -> Result<PathBuf, String> {
    if let Ok(value) = std::env::var("YA_DESKTOP_UV_PATH") {
        let path = PathBuf::from(value.trim());
        if path.exists() {
            return Ok(path);
        }
    }

    let uv_name = uv_binary_name();
    let managed_uv = layout.app_uv_dir.join(uv_name);
    if managed_uv.exists() {
        return Ok(managed_uv);
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        for candidate in [
            resource_dir.join("uv").join(uv_name),
            resource_dir.join("bin").join(uv_name),
            resource_dir.join(uv_name),
        ] {
            if candidate.exists() {
                fs::copy(&candidate, &managed_uv).map_err(|error| error.to_string())?;
                make_executable(&managed_uv)?;
                return Ok(managed_uv);
            }
        }
    }

    Ok(PathBuf::from(uv_name))
}

#[cfg(windows)]
fn uv_binary_name() -> &'static str {
    "uv.exe"
}

#[cfg(not(windows))]
fn uv_binary_name() -> &'static str {
    "uv"
}

#[cfg(windows)]
fn runtime_python_path(venv_dir: &Path) -> PathBuf {
    venv_dir.join("Scripts").join("python.exe")
}

#[cfg(not(windows))]
fn runtime_python_path(venv_dir: &Path) -> PathBuf {
    venv_dir.join("bin").join("python")
}

#[cfg(windows)]
fn runtime_clawd_path(venv_dir: &Path) -> PathBuf {
    venv_dir.join("Scripts").join("ya-clawd.exe")
}

#[cfg(not(windows))]
fn runtime_clawd_path(venv_dir: &Path) -> PathBuf {
    venv_dir.join("bin").join("ya-clawd")
}

fn make_executable(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)
            .map_err(|error| error.to_string())?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn run_uv_command(
    uv_path: &Path,
    layout: &RuntimeManagerLayout,
    args: &[String],
    log_file: &Path,
) -> Result<(), String> {
    append_log_line(
        log_file,
        &format!("$ {} {}", uv_path.display(), args.join(" ")),
    );
    let output = Command::new(uv_path)
        .args(args)
        .env("UV_CACHE_DIR", &layout.uv_cache_dir)
        .env("UV_PYTHON_INSTALL_DIR", &layout.uv_python_dir)
        .env("UV_LINK_MODE", "copy")
        .output()
        .map_err(|error| error.to_string())?;
    append_command_output(log_file, &output.stdout, &output.stderr);
    if output.status.success() {
        Ok(())
    } else {
        Err(format!("uv command failed with {}", output.status))
    }
}

fn run_runtime_command_capture(
    program: &Path,
    args: &[&str],
    log_file: &Path,
) -> Result<String, String> {
    append_log_line(
        log_file,
        &format!("$ {} {}", program.display(), args.join(" ")),
    );
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| error.to_string())?;
    append_command_output(log_file, &output.stdout, &output.stderr);
    if output.status.success() {
        String::from_utf8(output.stdout).map_err(|error| error.to_string())
    } else {
        Err(format!("runtime command failed with {}", output.status))
    }
}

fn run_runtime_command_with_env(
    program: &Path,
    args: &[&str],
    envs: &[(&str, &str)],
    log_file: &Path,
) -> Result<(), String> {
    append_log_line(
        log_file,
        &format!("$ {} {}", program.display(), args.join(" ")),
    );
    let mut command = Command::new(program);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().map_err(|error| error.to_string())?;
    append_command_output(log_file, &output.stdout, &output.stderr);
    if output.status.success() {
        Ok(())
    } else {
        Err(format!("runtime command failed with {}", output.status))
    }
}

fn append_command_output(log_file: &Path, stdout: &[u8], stderr: &[u8]) {
    if let Ok(text) = String::from_utf8(stdout.to_vec()) {
        for line in text.lines() {
            append_log_line(log_file, line);
        }
    }
    if let Ok(text) = String::from_utf8(stderr.to_vec()) {
        for line in text.lines() {
            append_log_line(log_file, line);
        }
    }
}

fn activate_claw_runtime(
    layout: &RuntimeManagerLayout,
    runtime: &ActiveClawRuntime,
) -> Result<(), String> {
    if layout.active_file.exists() {
        fs::copy(&layout.active_file, &layout.previous_active_file)
            .map_err(|error| error.to_string())?;
    }
    write_json_atomic(&layout.active_file, runtime)
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let tmp_path = path.with_extension("tmp");
    let content = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    fs::write(&tmp_path, content).map_err(|error| error.to_string())?;
    fs::rename(tmp_path, path).map_err(|error| error.to_string())
}

fn read_active_runtime(path: &Path) -> Result<ActiveClawRuntime, String> {
    let raw = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&raw).map_err(|error| error.to_string())
}

fn read_runtime_update_state(layout: &RuntimeManagerLayout) -> Result<RuntimeUpdateState, String> {
    if !layout.update_state_file.exists() {
        return Ok(RuntimeUpdateState::default());
    }
    let raw = fs::read_to_string(&layout.update_state_file).map_err(|error| error.to_string())?;
    serde_json::from_str(&raw).map_err(|error| error.to_string())
}

fn write_runtime_update_state(
    layout: &RuntimeManagerLayout,
    state: &RuntimeUpdateState,
) -> Result<(), String> {
    write_json_atomic(&layout.update_state_file, state)
}

fn list_installed_claw_runtimes(
    layout: &RuntimeManagerLayout,
    active: Option<&ActiveClawRuntime>,
) -> Result<Vec<InstalledClawRuntime>, String> {
    let mut runtimes = Vec::new();
    for entry in fs::read_dir(&layout.claw_dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if !path.is_dir() || path == layout.logs_dir {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let metadata = fs::read_to_string(path.join("runtime.json"))
            .ok()
            .and_then(|raw| serde_json::from_str::<ActiveClawRuntime>(&raw).ok());
        let active_runtime = active
            .map(|value| Path::new(&value.runtime_dir) == path)
            .unwrap_or(false);
        let log_file = layout.logs_dir.join(format!("{}.log", id));
        runtimes.push(InstalledClawRuntime {
            id,
            runtime_dir: path.to_string_lossy().to_string(),
            version: metadata.map(|value| value.version),
            active: active_runtime,
            failed: path.join("failed.json").exists(),
            log_file: log_file
                .exists()
                .then(|| log_file.to_string_lossy().to_string()),
        });
    }
    runtimes.sort_by(|left, right| right.id.cmp(&left.id));
    Ok(runtimes)
}

fn find_claw_runtime_dir(
    layout: &RuntimeManagerLayout,
    value: &str,
) -> Result<Option<PathBuf>, String> {
    let needle = value.trim();
    for runtime in list_installed_claw_runtimes(layout, None)? {
        if runtime.id == needle || runtime.version.as_deref() == Some(needle) {
            return Ok(Some(PathBuf::from(runtime.runtime_dir)));
        }
    }
    Ok(None)
}

fn sanitize_runtime_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn unix_timestamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())
        .map(|duration| duration.as_secs())
}

#[derive(Debug)]
struct LocalClawLayout {
    env_file: PathBuf,
    launch_config_file: PathBuf,
    profile_seed_file: PathBuf,
    data_dir: PathBuf,
    workspace_dir: PathBuf,
    sqlite_path: PathBuf,
    log_file: PathBuf,
    lock_file: PathBuf,
}

#[derive(Debug)]
struct ClawdCommandSpec {
    program: PathBuf,
    args: Vec<String>,
    display: String,
}

fn ensure_local_claw_layout(app: &AppHandle) -> Result<LocalClawLayout, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve app data dir: {}", error))?;
    let root = app_data_dir.join("local-claw");
    let data_dir = root.join("data");
    let workspace_dir = root.join("workspaces");
    let logs_dir = root.join("logs");
    fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&workspace_dir).map_err(|error| error.to_string())?;
    fs::create_dir_all(&logs_dir).map_err(|error| error.to_string())?;
    Ok(LocalClawLayout {
        env_file: root.join(".env"),
        launch_config_file: root.join("launch-config.json"),
        profile_seed_file: root.join("desktop-profiles.yaml"),
        sqlite_path: root.join("ya_claw.sqlite3"),
        lock_file: root.join("runtime.json"),
        log_file: logs_dir.join("ya-clawd.log"),
        data_dir,
        workspace_dir,
    })
}

fn desktop_workspace_status_from_layout(
    layout: &LocalClawLayout,
    config: &LocalClawLaunchConfig,
) -> Result<DesktopWorkspaceStatus, String> {
    let profile_seed_file = ensure_desktop_profile_seed_file(layout, config)?;
    Ok(DesktopWorkspaceStatus {
        workspace_root: layout.workspace_dir.to_string_lossy().to_string(),
        profile_seed_file: profile_seed_file.to_string_lossy().to_string(),
        relay_protocol: DESKTOP_RELAY_PROTOCOL.to_string(),
        shell_review_enabled: config.shell_review_enabled,
        shell_review_risk_threshold: config.shell_review_risk_threshold.clone(),
        shell_review_unattended_risk_threshold: config
            .shell_review_unattended_risk_threshold
            .clone(),
        shell_review_action: config.shell_review_action.clone(),
        shell_sandbox_enabled: config.shell_sandbox_enabled,
        shell_sandbox_backend: config.shell_sandbox_backend.clone(),
        shell_sandbox_network: config.shell_sandbox_network.clone(),
        shell_sandbox_allow_raw_host: config.shell_sandbox_allow_raw_host,
        message: "Desktop workspace is ready for local execution".to_string(),
    })
}

fn ensure_desktop_profile_seed_file(
    layout: &LocalClawLayout,
    config: &LocalClawLaunchConfig,
) -> Result<PathBuf, String> {
    let content = desktop_profile_seed_content(config)?;
    if let Some(parent) = layout.profile_seed_file.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&layout.profile_seed_file, content).map_err(|error| error.to_string())?;
    Ok(layout.profile_seed_file.clone())
}

fn desktop_profile_seed_content(config: &LocalClawLaunchConfig) -> Result<String, String> {
    let shell_action = validate_shell_review_action(&config.shell_review_action)?;
    let review_threshold = validate_shell_review_risk(&config.shell_review_risk_threshold)?;
    let unattended_threshold =
        validate_shell_review_risk(&config.shell_review_unattended_risk_threshold)?;
    let sandbox_backend = validate_shell_sandbox_backend(&config.shell_sandbox_backend)?;
    let sandbox_network = validate_shell_sandbox_network(&config.shell_sandbox_network)?;
    let model = yaml_string(&config.shell_review_model);
    let model_settings = yaml_string(&config.shell_review_model_settings);
    let enabled = bool_env(config.shell_review_enabled);
    let sandbox_enabled = bool_env(config.shell_sandbox_enabled);
    let sandbox_raw_host_approval = if config.shell_sandbox_allow_raw_host {
        "allowed_for_profile"
    } else {
        "requires_human"
    };
    Ok(format!(
        r#"version: 1
profiles:
- name: default
  model: gateway@openai-responses:gpt-5.5
  model_settings_preset: openai_responses_high
  model_config_preset: gpt5_270k
  security:
    shell_review:
      enabled: {enabled}
      model: {model}
      model_settings: {model_settings}
      on_needs_approval: {shell_action}
      risk_threshold: {review_threshold}
      unattended_risk_threshold: {unattended_threshold}
    shell_sandbox:
      enabled: {sandbox_enabled}
      profile: workspace_write
      backend_preference: {sandbox_backend}
      network: {sandbox_network}
      env_allowlist:
        - "*"
      raw_shell_approval: {sandbox_raw_host_approval}
      audit_enabled: true
  system_prompt: |-
    <agent_behavior>
    <identity>You are the YA Desktop workspace agent running through Local Claw.</identity>
    <workspace_contract>Use the selected Desktop Space as the execution boundary. Keep file and shell work inside the mounted workspace roots and preserve user project files carefully.</workspace_contract>
    <shell_safety>Shell commands are governed by Desktop shell review and Local Claw shell sandbox. Commands at or above the configured risk threshold enter human approval. Unattended work uses the stricter unattended threshold and denial mode for approval-required commands.</shell_safety>
    <relay_future>Desktop is the local capability host for {relay_protocol}; future central Claw agents can mount this device through explicit relay grants.</relay_future>
    </agent_behavior>
  builtin_toolsets:
  - core
  include_builtin_subagents: true
"#,
        relay_protocol = DESKTOP_RELAY_PROTOCOL,
    ))
}

fn validate_shell_review_action(value: &str) -> Result<&'static str, String> {
    match value.trim() {
        "defer" => Ok("defer"),
        "deny" => Ok("deny"),
        other => Err(format!("Invalid shell review action: {}", other)),
    }
}

fn validate_shell_review_risk(value: &str) -> Result<&'static str, String> {
    match value.trim() {
        "low" => Ok("low"),
        "medium" => Ok("medium"),
        "high" => Ok("high"),
        "extra_high" => Ok("extra_high"),
        other => Err(format!("Invalid shell review risk threshold: {}", other)),
    }
}

fn validate_shell_sandbox_backend(value: &str) -> Result<&'static str, String> {
    match value.trim() {
        "auto" => Ok("auto"),
        "linux_bwrap_seccomp" => Ok("linux_bwrap_seccomp"),
        "macos_seatbelt" => Ok("macos_seatbelt"),
        "windows_restricted_token" => Ok("windows_restricted_token"),
        "raw_host" => Ok("raw_host"),
        other => Err(format!("Invalid shell sandbox backend: {}", other)),
    }
}

fn validate_shell_sandbox_network(value: &str) -> Result<&'static str, String> {
    match value.trim() {
        "blocked" => Ok("blocked"),
        "restricted" => Ok("restricted"),
        "proxy" => Ok("proxy"),
        "full" => Ok("full"),
        other => Err(format!("Invalid shell sandbox network policy: {}", other)),
    }
}

fn yaml_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn ensure_local_api_token(env_file: &Path) -> Result<String, String> {
    if let Ok(content) = fs::read_to_string(env_file) {
        for line in content.lines() {
            if let Some(value) = line.strip_prefix("YA_CLAW_API_TOKEN=") {
                let token = value.trim().trim_matches('"').to_string();
                if !token.is_empty() {
                    harden_token_file_permissions(env_file)?;
                    return Ok(token);
                }
            }
        }
    }

    let token = generate_local_token()?;
    if let Some(parent) = env_file.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    write_token_file(env_file, &token)?;
    Ok(token)
}

fn read_local_claw_launch_config(config_file: &Path) -> Result<LocalClawLaunchConfig, String> {
    if !config_file.exists() {
        let config = with_config_file(LocalClawLaunchConfig::default(), config_file);
        write_local_claw_launch_config(config_file, &config)?;
        return Ok(config);
    }
    let raw = fs::read_to_string(config_file).map_err(|error| error.to_string())?;
    let config: LocalClawLaunchConfig =
        serde_json::from_str(&raw).map_err(|error| error.to_string())?;
    Ok(with_config_file(
        normalize_launch_config(config)?,
        config_file,
    ))
}

fn write_local_claw_launch_config(
    config_file: &Path,
    config: &LocalClawLaunchConfig,
) -> Result<(), String> {
    if let Some(parent) = config_file.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut persisted = config.clone();
    persisted.config_file = None;
    let raw = serde_json::to_string_pretty(&persisted).map_err(|error| error.to_string())?;
    fs::write(config_file, format!("{}\n", raw)).map_err(|error| error.to_string())?;
    Ok(())
}

fn with_config_file(
    mut config: LocalClawLaunchConfig,
    config_file: &Path,
) -> LocalClawLaunchConfig {
    config.config_file = Some(config_file.to_string_lossy().to_string());
    config
}

fn normalize_launch_config(
    mut config: LocalClawLaunchConfig,
) -> Result<LocalClawLaunchConfig, String> {
    let mut normalized_env = Vec::new();
    for entry in config.env {
        let key = entry.key.trim().to_string();
        let value = entry.value.trim().to_string();
        if key.is_empty() {
            continue;
        }
        validate_env_key(&key)?;
        validate_env_value(&value)?;
        match key.as_str() {
            "YA_CLAW_AGENCY_ENABLED" => {
                config.agency_enabled = parse_bool_env(&value)?;
            }
            "YA_CLAW_MEMORY_ENABLED" => {
                config.memory_enabled = parse_bool_env(&value)?;
            }
            "YA_CLAW_SHELL_SANDBOX_ENABLED" => {
                config.shell_sandbox_enabled = parse_bool_env(&value)?;
            }
            "YA_CLAW_SHELL_SANDBOX_BACKEND" => {
                config.shell_sandbox_backend = value;
            }
            "YA_CLAW_SHELL_SANDBOX_NETWORK" => {
                config.shell_sandbox_network = value;
            }
            "YA_CLAW_SHELL_SANDBOX_ALLOW_RAW_HOST" => {
                config.shell_sandbox_allow_raw_host = parse_bool_env(&value)?;
            }
            "YA_CLAW_API_TOKEN" | "YA_CLAW_PUBLIC_BASE_URL" | "YA_CLAW_PROFILE_SEED_FILE" => {
                return Err(format!(
                    "{} is managed by Desktop and cannot be overridden",
                    key
                ));
            }
            _ => normalized_env.push(LocalClawEnvVar { key, value }),
        }
    }
    validate_shell_review_action(&config.shell_review_action)?;
    validate_shell_review_risk(&config.shell_review_risk_threshold)?;
    validate_shell_review_risk(&config.shell_review_unattended_risk_threshold)?;
    validate_shell_sandbox_backend(&config.shell_sandbox_backend)?;
    validate_shell_sandbox_network(&config.shell_sandbox_network)?;
    if config.shell_review_enabled && config.shell_review_model.trim().is_empty() {
        return Err("Shell review model is required when shell review is enabled".to_string());
    }
    config.shell_review_model = config.shell_review_model.trim().to_string();
    config.shell_review_model_settings = config.shell_review_model_settings.trim().to_string();
    config.shell_review_risk_threshold = config.shell_review_risk_threshold.trim().to_string();
    config.shell_review_unattended_risk_threshold = config
        .shell_review_unattended_risk_threshold
        .trim()
        .to_string();
    config.shell_review_action = config.shell_review_action.trim().to_string();
    config.shell_sandbox_backend = config.shell_sandbox_backend.trim().to_string();
    config.shell_sandbox_network = config.shell_sandbox_network.trim().to_string();
    normalized_env.sort_by(|left, right| left.key.cmp(&right.key));
    normalized_env.dedup_by(|left, right| left.key == right.key);
    config.env = normalized_env;
    config.preset_name = config
        .preset_name
        .and_then(|value| {
            let trimmed = value.trim().to_string();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .or_else(|| Some("Desktop default".to_string()));
    config.config_file = None;
    Ok(config)
}

fn parse_launch_preset(raw: &str) -> Result<LocalClawLaunchConfig, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Launch preset cannot be empty".to_string());
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return parse_launch_preset_json(value);
    }
    parse_launch_preset_env(trimmed)
}

fn parse_launch_preset_json(value: serde_json::Value) -> Result<LocalClawLaunchConfig, String> {
    let Some(object) = value.as_object() else {
        return Err("Launch preset JSON must be an object".to_string());
    };
    let mut config = LocalClawLaunchConfig::default();
    if let Some(name) = object.get("name").and_then(|value| value.as_str()) {
        config.preset_name = Some(name.to_string());
    }
    if let Some(name) = object.get("presetName").and_then(|value| value.as_str()) {
        config.preset_name = Some(name.to_string());
    }
    if let Some(value) = object
        .get("agencyEnabled")
        .and_then(|value| value.as_bool())
    {
        config.agency_enabled = value;
    }
    if let Some(value) = object
        .get("memoryEnabled")
        .and_then(|value| value.as_bool())
    {
        config.memory_enabled = value;
    }
    if let Some(value) = object
        .get("shellReviewEnabled")
        .and_then(|value| value.as_bool())
    {
        config.shell_review_enabled = value;
    }
    if let Some(value) = object
        .get("shellReviewModel")
        .and_then(|value| value.as_str())
    {
        config.shell_review_model = value.to_string();
    }
    if let Some(value) = object
        .get("shellReviewModelSettings")
        .and_then(|value| value.as_str())
    {
        config.shell_review_model_settings = value.to_string();
    }
    if let Some(value) = object
        .get("shellReviewRiskThreshold")
        .and_then(|value| value.as_str())
    {
        config.shell_review_risk_threshold = value.to_string();
    }
    if let Some(value) = object
        .get("shellReviewUnattendedRiskThreshold")
        .and_then(|value| value.as_str())
    {
        config.shell_review_unattended_risk_threshold = value.to_string();
    }
    if let Some(value) = object
        .get("shellReviewAction")
        .and_then(|value| value.as_str())
    {
        config.shell_review_action = value.to_string();
    }
    if let Some(value) = object
        .get("shellSandboxEnabled")
        .and_then(|value| value.as_bool())
    {
        config.shell_sandbox_enabled = value;
    }
    if let Some(value) = object
        .get("shellSandboxBackend")
        .and_then(|value| value.as_str())
    {
        config.shell_sandbox_backend = value.to_string();
    }
    if let Some(value) = object
        .get("shellSandboxNetwork")
        .and_then(|value| value.as_str())
    {
        config.shell_sandbox_network = value.to_string();
    }
    if let Some(value) = object
        .get("shellSandboxAllowRawHost")
        .and_then(|value| value.as_bool())
    {
        config.shell_sandbox_allow_raw_host = value;
    }
    for key in ["env", "environment", "variables"] {
        if let Some(env_value) = object.get(key) {
            append_json_env(env_value, &mut config.env)?;
        }
    }
    normalize_launch_config(config)
}

fn append_json_env(
    value: &serde_json::Value,
    env: &mut Vec<LocalClawEnvVar>,
) -> Result<(), String> {
    if let Some(object) = value.as_object() {
        let ordered: BTreeMap<_, _> = object.iter().collect();
        for (key, value) in ordered {
            env.push(LocalClawEnvVar {
                key: key.to_string(),
                value: json_env_value(value)?,
            });
        }
        return Ok(());
    }
    if let Some(items) = value.as_array() {
        for item in items {
            let Some(object) = item.as_object() else {
                return Err("Launch preset env array items must be objects".to_string());
            };
            let key = object
                .get("key")
                .and_then(|value| value.as_str())
                .ok_or_else(|| "Launch preset env item is missing key".to_string())?;
            let value = object
                .get("value")
                .map(json_env_value)
                .transpose()?
                .unwrap_or_default();
            env.push(LocalClawEnvVar {
                key: key.to_string(),
                value,
            });
        }
        return Ok(());
    }
    Err("Launch preset env must be an object or array".to_string())
}

fn json_env_value(value: &serde_json::Value) -> Result<String, String> {
    if let Some(string) = value.as_str() {
        return Ok(string.to_string());
    }
    if value.is_boolean() || value.is_number() {
        return Ok(value.to_string());
    }
    Err("Launch preset env values must be strings, booleans, or numbers".to_string())
}

fn parse_launch_preset_env(raw: &str) -> Result<LocalClawLaunchConfig, String> {
    let mut config = LocalClawLaunchConfig {
        preset_name: Some("Imported env preset".to_string()),
        ..Default::default()
    };
    let mut saw_env = false;
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("Invalid preset line: {}", line));
        };
        saw_env = true;
        config.env.push(LocalClawEnvVar {
            key: key.trim().to_string(),
            value: value.trim().trim_matches('"').to_string(),
        });
    }
    if !saw_env {
        return Err("Launch preset did not contain any environment variables".to_string());
    }
    normalize_launch_config(config)
}

fn validate_env_key(key: &str) -> Result<(), String> {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return Err("Environment variable key cannot be empty".to_string());
    };
    if !(first == '_' || first.is_ascii_uppercase()) {
        return Err(format!(
            "Environment variable key must start with A-Z or _: {}",
            key
        ));
    }
    if !chars.all(|ch| ch == '_' || ch.is_ascii_uppercase() || ch.is_ascii_digit()) {
        return Err(format!(
            "Environment variable key must contain only A-Z, 0-9, and _: {}",
            key
        ));
    }
    Ok(())
}

fn validate_env_value(value: &str) -> Result<(), String> {
    if value.contains('\0') || value.contains('\n') || value.contains('\r') {
        return Err("Environment variable values cannot contain NUL or newlines".to_string());
    }
    Ok(())
}

fn parse_bool_env(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("Invalid boolean environment value: {}", value)),
    }
}

fn bool_env(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn reserve_local_claw_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("Failed to reserve a Local Claw port: {}", error))?;
    let port = listener
        .local_addr()
        .map_err(|error| format!("Failed to inspect reserved Local Claw port: {}", error))?
        .port();
    drop(listener);
    Ok(port)
}

fn generate_local_token() -> Result<String, String> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| error.to_string())?;
    Ok(format!("desktop-local-{}", hex::encode(bytes)))
}

fn write_token_file(env_file: &Path, token: &str) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(env_file)
            .map_err(|error| error.to_string())?;
        writeln!(file, "YA_CLAW_API_TOKEN={}", token).map_err(|error| error.to_string())?;
        harden_token_file_permissions(env_file)?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        fs::write(env_file, format!("YA_CLAW_API_TOKEN={}\n", token))
            .map_err(|error| error.to_string())
    }
}

fn harden_token_file_permissions(env_file: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(env_file)
            .map_err(|error| error.to_string())?
            .permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(env_file, permissions).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn resolve_clawd_command(app: &AppHandle) -> Result<ClawdCommandSpec, String> {
    if let Ok(program) = std::env::var("YA_DESKTOP_CLAWD_PROGRAM") {
        let program = program.trim();
        if !program.is_empty() {
            let args = std::env::var("YA_DESKTOP_CLAWD_ARGS")
                .ok()
                .map(|raw| split_command_args(&raw))
                .transpose()?
                .unwrap_or_default();
            return Ok(ClawdCommandSpec {
                program: PathBuf::from(program),
                args,
                display: format!("{} [YA_DESKTOP_CLAWD_ARGS]", program),
            });
        }
    }

    if let Ok(raw_command) = std::env::var("YA_DESKTOP_CLAWD_COMMAND") {
        if !raw_command.trim().is_empty() {
            let parts = split_command_args(&raw_command)?;
            if let Some((program, args)) = parts.split_first() {
                return Ok(ClawdCommandSpec {
                    program: PathBuf::from(program),
                    args: args.to_vec(),
                    display: raw_command,
                });
            }
        }
    }

    if let Some(active_runtime) = resolve_active_claw_runtime(app) {
        return Ok(ClawdCommandSpec {
            display: active_runtime.to_string_lossy().to_string(),
            program: active_runtime,
            args: Vec::new(),
        });
    }

    Ok(ClawdCommandSpec {
        program: PathBuf::from("uv"),
        args: vec![
            "run".to_string(),
            "--package".to_string(),
            "ya-claw".to_string(),
            "ya-clawd".to_string(),
        ],
        display: "uv run --package ya-claw ya-clawd".to_string(),
    })
}

fn resolve_active_claw_runtime(app: &AppHandle) -> Option<PathBuf> {
    let active_file = app
        .path()
        .app_data_dir()
        .ok()?
        .join("runtimes")
        .join("claw")
        .join("active.json");
    let raw = fs::read_to_string(active_file).ok()?;
    let metadata: ActiveClawRuntime = serde_json::from_str(&raw).ok()?;
    let entrypoint = PathBuf::from(metadata.entrypoint);
    entrypoint.exists().then_some(entrypoint)
}

fn resolve_profile_seed_file(_app: &AppHandle) -> Option<PathBuf> {
    None
}

fn wait_for_ready(
    child: &mut Child,
    stdout: impl std::io::Read + Send + 'static,
    stderr: impl std::io::Read + Send + 'static,
    mut base_info: LocalClawRuntimeInfo,
    log_file: PathBuf,
) -> Result<LocalClawRuntimeInfo, String> {
    let (sender, receiver) = mpsc::channel::<ProcessLine>();
    spawn_line_reader(
        stdout,
        ProcessLine::Stdout,
        sender.clone(),
        log_file.clone(),
    );
    spawn_line_reader(stderr, ProcessLine::Stderr, sender, log_file.clone());

    let deadline = Instant::now() + LOCAL_CLAW_READY_TIMEOUT;
    let mut last_stderr: Option<String> = None;

    loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            let _ = child.wait();
            return Err(format!(
                "Local Claw exited before readiness: {}; see {}{}",
                status,
                log_file.display(),
                last_stderr
                    .as_ref()
                    .map(|line| format!("; last stderr: {}", line))
                    .unwrap_or_default()
            ));
        }

        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Timed out waiting for Local Claw readiness after {}s; see {}{}",
                LOCAL_CLAW_READY_TIMEOUT.as_secs(),
                log_file.display(),
                last_stderr
                    .as_ref()
                    .map(|line| format!("; last stderr: {}", line))
                    .unwrap_or_default()
            ));
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        let poll_timeout = remaining.min(Duration::from_millis(250));
        match receiver.recv_timeout(poll_timeout) {
            Ok(ProcessLine::Stdout(line)) => {
                if let Ok(ready) = serde_json::from_str::<ReadyLine>(&line) {
                    if ready.event_type == "ya_clawd.ready" {
                        base_info.base_url = ready.base_url;
                        base_info.pid = ready.pid;
                        if let Some(data_dir) = ready.data_dir {
                            base_info.data_dir = data_dir;
                        }
                        if let Some(workspace_dir) = ready.workspace_dir {
                            base_info.workspace_dir = workspace_dir;
                        }
                        return Ok(base_info);
                    }
                }
            }
            Ok(ProcessLine::Stderr(line)) => {
                last_stderr = Some(line);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {}
        }
    }
}

fn spawn_line_reader<R, F>(
    reader: R,
    constructor: F,
    sender: mpsc::Sender<ProcessLine>,
    log_file: PathBuf,
) where
    R: std::io::Read + Send + 'static,
    F: Fn(String) -> ProcessLine + Send + 'static,
{
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            append_log_line(&log_file, &line);
            if sender.send(constructor(line)).is_err() {
                break;
            }
        }
    });
}

fn append_log_line(log_file: &Path, line: &str) {
    if let Some(parent) = log_file.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_file) {
        let _ = writeln!(file, "{}", line);
    }
}

fn split_command_args(command: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut quote: Option<char> = None;

    while let Some(ch) = chars.next() {
        match (ch, quote) {
            ('\\', _) => {
                if let Some(next) = chars.next() {
                    current.push(next);
                } else {
                    current.push(ch);
                }
            }
            ('\'' | '"', None) => quote = Some(ch),
            (value, Some(active_quote)) if value == active_quote => quote = None,
            (value, None) if value.is_whitespace() => {
                if !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }

    if let Some(active_quote) = quote {
        return Err(format!(
            "Unclosed quote {} in command override",
            active_quote
        ));
    }
    if !current.is_empty() {
        args.push(current);
    }
    Ok(args)
}

fn status_from_info(info: &LocalClawRuntimeInfo, running: bool, message: &str) -> LocalClawStatus {
    let api_token = if running {
        PathBuf::from(&info.data_dir)
            .parent()
            .and_then(|root| ensure_local_api_token(&root.join(".env")).ok())
    } else {
        None
    };
    LocalClawStatus {
        running,
        base_url: running.then(|| info.base_url.clone()),
        pid: running.then_some(info.pid),
        data_dir: Some(info.data_dir.clone()),
        workspace_dir: Some(info.workspace_dir.clone()),
        sqlite_path: Some(info.sqlite_path.clone()),
        log_file: Some(info.log_file.clone()),
        lock_file: Some(info.lock_file.clone()),
        api_token,
        profile_seed_file: PathBuf::from(&info.data_dir).parent().map(|root| {
            root.join("desktop-profiles.yaml")
                .to_string_lossy()
                .to_string()
        }),
        relay_protocol: DESKTOP_RELAY_PROTOCOL.to_string(),
        message: message.to_string(),
    }
}

fn stopped_status(message: &str) -> LocalClawStatus {
    LocalClawStatus {
        running: false,
        base_url: None,
        pid: None,
        data_dir: None,
        workspace_dir: None,
        sqlite_path: None,
        log_file: None,
        lock_file: None,
        api_token: None,
        profile_seed_file: None,
        relay_protocol: DESKTOP_RELAY_PROTOCOL.to_string(),
        message: message.to_string(),
    }
}

pub fn run() {
    tauri::Builder::default()
        .manage(LocalClawManager::default())
        .setup(|app| {
            let app_handle = app.handle().clone();
            thread::spawn(move || {
                let state = app_handle.state::<LocalClawManager>();
                if let Err(error) = start_local_claw(app_handle.clone(), state) {
                    eprintln!("Failed to auto-start Local Claw sidecar: {}", error);
                }

                thread::sleep(CLAW_AUTO_UPDATE_INITIAL_DELAY);
                let runtime_manager = app_handle.state::<RuntimeManager>();
                if let Err(error) =
                    check_claw_runtime_update_inner(app_handle.clone(), runtime_manager, false)
                {
                    eprintln!("Failed to check Claw runtime update: {}", error);
                }
            });
            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(RuntimeManager::default())
        .invoke_handler(tauri::generate_handler![
            get_local_claw_status,
            start_local_claw,
            stop_local_claw,
            restart_local_claw,
            get_local_claw_launch_config,
            update_local_claw_launch_config,
            reset_local_claw_launch_config,
            import_local_claw_launch_preset,
            get_desktop_workspace_status,
            run_desktop_onboarding,
            get_runtime_manager_status,
            install_latest_claw_runtime,
            update_claw_runtime,
            repair_claw_runtime,
            remove_claw_runtime,
            check_claw_runtime_update,
            apply_ready_claw_runtime_update,
            get_runtime_install_log,
        ])
        .build(tauri::generate_context!())
        .expect("error while building YA Desktop")
        .run(|app_handle, event| {
            if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
                let state = app_handle.state::<LocalClawManager>();
                let _ = stop_local_claw_with_message(
                    state,
                    "Local Claw sidecar stopped on desktop exit",
                );
            }
        });
}
