#[tauri::command]
fn get_local_claw_status() -> LocalClawStatus {
    LocalClawStatus {
        running: false,
        base_url: None,
        message: "Local Claw sidecar is not wired yet".to_string(),
    }
}

#[derive(serde::Serialize)]
struct LocalClawStatus {
    running: bool,
    base_url: Option<String>,
    message: String,
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![get_local_claw_status])
        .run(tauri::generate_context!())
        .expect("error while running YA Desktop");
}
