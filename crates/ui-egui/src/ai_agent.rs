//! AI Agent pane & runtime:
//! - Cloud LLM API (OpenRouter, AgentRouter, OpenAI, Anthropic, Gemini, DeepSeek, Groq, Custom)
//!   with automatic model routing (user only enters API Key).
//! - Terminal CLI Agents (`agy`, `claude`, `codex`, custom CLI) with model selection and
//!   per-user account model access verification.
//! - Local LLM via Ollama (`http://localhost:11434`) with automatic detection of installed models.
//!
//! Each mode maintains its own independent connection status and background worker so switching
//! tabs NEVER disconnects or stops an active connection in another tab!
//!
//! Fork-local feature: the pane and its heuristics predate the workspace's strict Clippy policy
//! (0.13), and the upstream sync keeps them intact rather than reshaping ~6k lines of working AI
//! logic. Split the long functions and argument lists when the pane is reworked.
#![allow(clippy::cognitive_complexity, clippy::too_many_arguments)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use egui::{Color32, RichText, Stroke, Ui, vec2};
use gridcraft_engine::core::{CellRef, RangeRef, col_to_letters, split_sheet};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::SheetApp;
use crate::control::{self, ControlRequest, Outcome};
use crate::theme::{self, Tokens};

/// Tab selector in the AI Agent pane. Switching tabs only changes the view — active connections
/// in other tabs stay connected!
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMode {
    Api,
    Terminal,
    Ollama,
}

impl AgentMode {
    pub fn short_label(self) -> &'static str {
        match self {
            AgentMode::Api => "LLM API",
            AgentMode::Terminal => "Terminal Agent",
            AgentMode::Ollama => "Ollama (Local)",
        }
    }
}

/// Supported Cloud API providers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiProvider {
    OpenRouter,
    AgentRouter,
    OpenAI,
    Anthropic,
    Gemini,
    DeepSeek,
    Groq,
    Custom,
}

impl ApiProvider {
    pub const ALL: &'static [ApiProvider] = &[
        ApiProvider::OpenRouter,
        ApiProvider::AgentRouter,
        ApiProvider::OpenAI,
        ApiProvider::Anthropic,
        ApiProvider::Gemini,
        ApiProvider::DeepSeek,
        ApiProvider::Groq,
        ApiProvider::Custom,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ApiProvider::OpenRouter => "OpenRouter",
            ApiProvider::AgentRouter => "AgentRouter",
            ApiProvider::OpenAI => "OpenAI",
            ApiProvider::Anthropic => "Anthropic (Claude)",
            ApiProvider::Gemini => "Google Gemini",
            ApiProvider::DeepSeek => "DeepSeek",
            ApiProvider::Groq => "Groq",
            ApiProvider::Custom => "Custom (OpenAI-Compatible)",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            ApiProvider::OpenRouter => "https://openrouter.ai/api/v1",
            ApiProvider::AgentRouter => "https://agentrouter.org/v1",
            ApiProvider::OpenAI => "https://api.openai.com/v1",
            ApiProvider::Anthropic => "https://api.anthropic.com/v1",
            ApiProvider::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            ApiProvider::DeepSeek => "https://api.deepseek.com/v1",
            ApiProvider::Groq => "https://api.groq.com/openai/v1",
            ApiProvider::Custom => "https://openrouter.ai/api/v1",
        }
    }

    /// Prioritized candidate models for automatic selection when user connects with an API Key (2026 generation).
    pub fn candidate_models(self) -> &'static [&'static str] {
        match self {
            ApiProvider::OpenRouter => &[
                "openrouter/auto",
                "google/gemini-3.8-flash",
                "google/gemini-3.1-pro",
                "anthropic/claude-sonnet-5-5",
                "anthropic/claude-opus-4-8",
                "openai/gpt-5.5",
                "openai/gpt-5.4-mini",
                "deepseek/deepseek-flash",
                "openai/gpt-oss-120b:free",
            ],
            ApiProvider::AgentRouter => {
                &["claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-4-8", "gpt-5.5", "gpt-5.4-mini", "gemini-3.8-flash", "deepseek-flash", "auto"]
            }
            ApiProvider::OpenAI => &["gpt-5.5", "gpt-5.4-mini", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra"],
            ApiProvider::Anthropic => &[
                "claude-sonnet-5-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-5-5",
                "claude-fable-5-1",
                "claude-fable-5",
                "claude-haiku-5-5",
                "claude-sonnet-4-6",
            ],
            ApiProvider::Gemini => &["gemini-3.8-flash", "gemini-4-argon", "gemini-3.1-pro", "gemini-3.5-flash", "gemini-3.1-flash-lite"],
            ApiProvider::DeepSeek => &["deepseek-chat", "deepseek-reasoner", "deepseek-flash", "deepseek-v4-pro"],
            ApiProvider::Groq => &["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            ApiProvider::Custom => &["gpt-5.5", "auto"],
        }
    }

    /// Parse a provider name/id as used by agents, the control channel and tests: `"deepseek"`,
    /// `"DeepSeek"`, `"openai"`, `"claude"`, `"custom"`…
    pub fn from_str_name(name: &str) -> Option<ApiProvider> {
        let wanted = name.trim().to_ascii_lowercase();
        if wanted.is_empty() {
            return None;
        }
        ApiProvider::ALL.iter().copied().find(|p| {
            let variant = format!("{p:?}").to_ascii_lowercase();
            variant == wanted
                || p.label().to_ascii_lowercase() == wanted
                || variant.contains(&wanted)
                || p.label().to_ascii_lowercase().contains(&wanted)
        })
    }
}

/// Models that can answer chat turns: drop embedding, speech and moderation endpoints and
/// duplicates, keeping the provider's order.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn filter_chat_models(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for id in ids {
        let id = id.trim().to_string();
        if id.is_empty() {
            continue;
        }
        let lower = id.to_ascii_lowercase();
        if lower.contains("embed") || lower.contains("whisper") || lower.contains("tts") || lower.contains("moderation") {
            continue;
        }
        if !out.iter().any(|existing| existing == &id) {
            out.push(id);
        }
    }
    out
}

/// Merge the provider's reported models with the built-in candidates: the provider's order wins,
/// candidates fill the gaps, duplicates are dropped.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn merge_api_models(listed: Vec<String>, candidates: &[&str]) -> Vec<String> {
    let mut all = filter_chat_models(listed);
    for candidate in candidates {
        let candidate = (*candidate).to_string();
        if !all.iter().any(|existing| existing == &candidate) {
            all.push(candidate);
        }
    }
    all
}

/// Order to use models in: the user's pick first, then the models the provider reports for this
/// key (authoritative), then the built-in candidates. De-duplicated, empty entries dropped.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn build_api_candidate_order(preferred: Option<&str>, discovered: &[String], candidates: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(pick) = preferred.map(str::trim).filter(|pick| !pick.is_empty()) {
        out.push(pick.to_string());
    }
    for model in discovered {
        let model = model.trim();
        if !model.is_empty() && !out.iter().any(|existing| existing == model) {
            out.push(model.to_string());
        }
    }
    for candidate in candidates {
        if !out.iter().any(|existing| existing == *candidate) {
            out.push((*candidate).to_string());
        }
    }
    out
}

/// Supported Terminal CLI Agents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CliAgent {
    Agy,
    Claude,
    Codex,
    Custom,
}

impl CliAgent {
    pub const ALL: &'static [CliAgent] = &[CliAgent::Agy, CliAgent::Claude, CliAgent::Codex, CliAgent::Custom];

    pub fn label(self) -> &'static str {
        match self {
            CliAgent::Agy => "Antigravity CLI (agy)",
            CliAgent::Claude => "Claude Code (claude)",
            CliAgent::Codex => "OpenAI Codex (codex)",
            CliAgent::Custom => "Custom Terminal Command",
        }
    }

    pub fn cmd_name(self) -> &'static str {
        match self {
            CliAgent::Agy => "agy",
            CliAgent::Claude => "claude",
            CliAgent::Codex => "codex",
            CliAgent::Custom => "",
        }
    }

    pub fn default_models(self) -> Vec<CliModelInfo> {
        #[cfg(not(target_arch = "wasm32"))]
        if self == CliAgent::Codex
            && let Some(cached) = discover_codex_models_from_disk()
            && !cached.is_empty()
        {
            return cached.into_iter().map(|(id, name)| CliModelInfo { id, display_name: name, access: CliModelAccess::Unknown }).collect();
        }

        let pairs: &[(&str, &str)] = match self {
            CliAgent::Agy => &[
                ("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"),
                ("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)"),
                ("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"),
                ("gemini-4-argon-low", "Gemini 4 Argon (Low)"),
                ("gemini-4-argon-medium", "Gemini 4 Argon (Medium)"),
                ("gemini-4-argon-high", "Gemini 4 Argon (High)"),
                ("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"),
                ("claude-sonnet-5-5-low", "Claude Sonnet 5.5 (Low)"),
                ("claude-sonnet-5-5-medium", "Claude Sonnet 5.5 (Medium)"),
                ("claude-sonnet-5-5-high", "Claude Sonnet 5.5 (High)"),
                ("claude-opus-5-5-low", "Claude Opus 5.5 (Low)"),
                ("claude-opus-5-5-medium", "Claude Opus 5.5 (Medium)"),
                ("claude-opus-5-5-high", "Claude Opus 5.5 (High)"),
            ],
            CliAgent::Claude => &[
                ("sonnet", "Claude Sonnet 5 (Default 1M)"),
                ("opus", "Claude Opus 4.8 / 5.5"),
                ("opus[1m]", "Claude Opus (1M Context)"),
                ("fable", "Claude Fable 5"),
                ("haiku", "Claude Haiku 5.5"),
                ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
                ("claude-opus-5-5", "Claude Opus 5.5"),
                ("claude-opus-4-8", "Claude Opus 4.8"),
                ("claude-fable-5", "Claude Fable 5 (Full ID)"),
            ],
            CliAgent::Codex => &[
                ("gpt-5.5", "GPT-5.5 (Frontier Coding & Reasoning)"),
                ("gpt-5.4-mini", "GPT-5.4-Mini (Fast & Cost-Efficient)"),
                ("gpt-6-sol", "GPT-6 Sol"),
                ("gpt-6-astra", "GPT-6 Astra"),
            ],
            CliAgent::Custom => &[("default", "Default CLI Model")],
        };
        pairs
            .iter()
            .map(|(id, name)| CliModelInfo { id: (*id).to_string(), display_name: (*name).to_string(), access: CliModelAccess::Unknown })
            .collect()
    }
}

/// Per-user verification status of a Terminal Agent model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CliModelAccess {
    Unknown,
    Usable,
    Unavailable(String),
}

/// Information and verification status for a single Terminal Agent model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliModelInfo {
    pub id: String,
    pub display_name: String,
    pub access: CliModelAccess,
}

impl CliModelInfo {
    pub fn badge_label(&self) -> String {
        match &self.access {
            CliModelAccess::Usable => format!("OK {} ({})", self.display_name, self.id),
            CliModelAccess::Unavailable(_) => format!("X {} ({}) [Unavailable]", self.display_name, self.id),
            CliModelAccess::Unknown => format!("• {} ({})", self.display_name, self.id),
        }
    }
}

/// Connection indicator status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnStatus {
    Disconnected,
    Connecting,
    Connected(String),
    Failed(String),
}

impl ConnStatus {
    pub fn is_connected(&self) -> bool {
        matches!(self, ConnStatus::Connected(_))
    }
    pub fn is_connecting(&self) -> bool {
        matches!(self, ConnStatus::Connecting)
    }
}

/// Chat history item.
#[derive(Clone, Debug)]
pub enum ChatEntry {
    User(String),
    Assistant(String),
    ToolCall { name: String, summary: String, is_error: bool },
    System(String),
}

/// Events sent from background worker threads back to the UI.
pub enum WorkerEvent {
    ConnResult {
        mode: AgentMode,
        result: Result<String, String>,
        resolved_model: Option<String>,
        models: Option<Vec<String>>,
    },
    /// Models reported by the cloud provider's `/models` endpoint (or an error from trying).
    ApiModelsListed {
        models: Vec<String>,
        error: Option<String>,
    },
    OllamaModelsScanned {
        models: Vec<String>,
        error: Option<String>,
    },
    CliModelsDiscovered {
        agent: CliAgent,
        models: Vec<(String, String)>,
    },
    CliModelVerified {
        agent: CliAgent,
        model_id: String,
        usable: Result<(), String>,
    },
    CliScanFinished,
    ToolStart {
        name: String,
        args: String,
    },
    ToolDone {
        name: String,
        summary: String,
        is_error: bool,
    },
    AssistantReply(String),
    Error(String),
    ConnectionLost {
        mode: AgentMode,
        reason: String,
    },
    Finished,
}

/// State of the AI Agent panel.
pub struct AiAgentState {
    /// Currently viewed configuration tab (`Api`, `Terminal`, `Ollama`).
    /// Switching this NEVER disconnects active connections!
    pub mode: AgentMode,
    /// Which connected backend is selected to handle chat messages when multiple are connected.
    pub active_chat_backend: AgentMode,

    /// Give the agent the whole command surface: catalog-aware discovery tools, arbitrary-command
    /// execution (`execute_command`/`execute_commands`), selection/sort/undo helpers, and no
    /// hard-coded single-purpose shortcuts in place of the commands the model asked for.
    pub full_tool_access: bool,
    /// Put the compact live command catalog in the model prompt (needs more context tokens but makes
    /// every feature discoverable without a first `list_commands` round trip).
    pub include_command_catalog: bool,
    /// Allow the `ui_control` tool (inspect UI state, switch ribbon tabs/theme, interface language,
    /// read-only workbook queries; dialogs and pointer input stay blocked).
    pub allow_ui_tools: bool,

    // --- Cloud LLM API config & independent connection state ---
    pub provider: ApiProvider,
    pub base_url: String,
    pub api_key: String,
    pub show_api_key: bool,
    /// Model used for the next API chat turn. Picked from the dropdown, typed by hand, or resolved
    /// automatically when the connection is verified.
    pub model: String,
    /// Models the API key can use: the provider's `/models` list first, then the built-in
    /// candidates. Drives the Model dropdown.
    pub api_models: Vec<String>,
    pub api_models_scanning: bool,
    pub api_models_error: Option<String>,
    pub api_models_rx: Option<Receiver<WorkerEvent>>,
    pub api_models_cancel: Arc<AtomicBool>,
    pub api_status: ConnStatus,
    pub api_verify_rx: Option<Receiver<WorkerEvent>>,
    pub api_verify_cancel: Arc<AtomicBool>,

    // --- Terminal CLI Agent config & independent connection state ---
    pub cli_agent: CliAgent,
    pub custom_cli_cmd: String,
    pub cli_model: String,
    pub cli_models: Vec<CliModelInfo>,
    pub cli_scanning_models: bool,
    pub cli_status: ConnStatus,
    pub cli_verify_rx: Option<Receiver<WorkerEvent>>,
    pub cli_verify_cancel: Arc<AtomicBool>,

    // --- Local LLM (Ollama) config & independent connection state ---
    pub ollama_url: String,
    pub ollama_model: String,
    /// Strictly the models actually installed in the user's local Ollama instance.
    pub ollama_models: Vec<String>,
    pub ollama_scanned: bool,
    pub ollama_scanning: bool,
    pub ollama_scan_error: Option<String>,
    pub ollama_status: ConnStatus,
    pub ollama_verify_rx: Option<Receiver<WorkerEvent>>,
    pub ollama_verify_cancel: Arc<AtomicBool>,

    // --- Shared Chat & Active Task state ---
    pub messages: Vec<ChatEntry>,
    pub chat_input: String,
    pub running: bool,
    pub cancel_flag: Arc<AtomicBool>,
    pub worker_rx: Option<Receiver<WorkerEvent>>,
    /// Dismissible English error notification banner shown when a process fails.
    pub error_notification: Option<String>,
    /// English toast notification forwarded to `SheetApp::toast`.
    pub pending_toast: Option<String>,

    /// Dedicated in-process control channel so background AI threads can execute MCP tools on `SheetApp`
    /// even when `--control` TCP server is also active.
    pub control_tx: Option<Sender<ControlRequest>>,
    pub control_rx: Option<Receiver<ControlRequest>>,
    /// Local TCP control port if the desktop host started a control listener.
    pub control_port: Option<u16>,
    /// Snapshot of Table 1's style/formula/table blueprint before the turn starts so merged tables match Table 1!
    #[cfg(not(target_arch = "wasm32"))]
    table_blueprint: Option<TableStyleBlueprint>,
}

impl Default for AiAgentState {
    fn default() -> Self {
        let provider = ApiProvider::OpenRouter;
        let cli_agent = CliAgent::Agy;
        let cli_models = cli_agent.default_models();
        let default_cli_model = cli_models.first().map(|m| m.id.clone()).unwrap_or_else(|| "gemini-3.8-flash-low".to_string());
        let (control_tx, control_rx) = channel();

        Self {
            mode: AgentMode::Api,
            active_chat_backend: AgentMode::Api,

            full_tool_access: true,
            include_command_catalog: true,
            allow_ui_tools: true,

            provider,
            base_url: provider.default_base_url().to_string(),
            api_key: String::new(),
            show_api_key: false,
            model: provider.candidate_models().first().copied().unwrap_or("openrouter/auto").to_string(),
            api_models: provider.candidate_models().iter().map(|s| (*s).to_string()).collect(),
            api_models_scanning: false,
            api_models_error: None,
            api_models_rx: None,
            api_models_cancel: Arc::new(AtomicBool::new(false)),
            api_status: ConnStatus::Disconnected,
            api_verify_rx: None,
            api_verify_cancel: Arc::new(AtomicBool::new(false)),

            cli_agent,
            custom_cli_cmd: cli_agent.cmd_name().to_string(),
            cli_model: default_cli_model,
            cli_models,
            cli_scanning_models: false,
            cli_status: ConnStatus::Disconnected,
            cli_verify_rx: None,
            cli_verify_cancel: Arc::new(AtomicBool::new(false)),

            ollama_url: "http://127.0.0.1:11434".to_string(),
            ollama_model: String::new(),
            ollama_models: Vec::new(),
            ollama_scanned: false,
            ollama_scanning: false,
            ollama_scan_error: None,
            ollama_status: ConnStatus::Disconnected,
            ollama_verify_rx: None,
            ollama_verify_cancel: Arc::new(AtomicBool::new(false)),

            messages: Vec::new(),
            chat_input: String::new(),
            running: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            worker_rx: None,
            error_notification: None,
            pending_toast: None,
            control_tx: Some(control_tx),
            control_rx: Some(control_rx),
            control_port: None,
            #[cfg(not(target_arch = "wasm32"))]
            table_blueprint: None,
        }
    }
}

/// Poll both in-process AI control requests and AI worker events each frame.
pub fn poll_app(app: &mut SheetApp, ctx: &egui::Context) {
    if let Some(rx) = app.ai_agent.control_rx.take() {
        while let Ok(req) = rx.try_recv() {
            if !req.take() {
                continue;
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(custom_reply) = handle_ai_custom_control(app, &req.method, &req.params) {
                let _ = req.reply.send(custom_reply);
                continue;
            }
            let outcome = control::handle(app, ctx, &req.method, &req.params);
            match outcome {
                Outcome::Done(v) => {
                    let _ = req.reply.send(v);
                }
                Outcome::Screenshot { path } => {
                    let _ = req.reply.send(json!({"ok": true, "result": {"path": path}}));
                }
            }
        }
        app.ai_agent.control_rx = Some(rx);
    }
    app.ai_agent.poll(ctx);
    if let Some(toast_msg) = app.ai_agent.pending_toast.take() {
        app.toast = Some((toast_msg, crate::now_ms()));
    }
}

impl AiAgentState {
    /// Switch Cloud API provider and populate default URL & candidate model (only resets Cloud API status).
    pub fn set_provider(&mut self, p: ApiProvider) {
        if self.provider != p {
            self.provider = p;
            self.base_url = p.default_base_url().to_string();
            self.api_models = p.candidate_models().iter().map(|s| (*s).to_string()).collect();
            self.api_models_error = None;
            self.api_models_scanning = false;
            self.model = self.api_models.first().cloned().unwrap_or_else(|| "gpt-5.5".to_string());
            self.disconnect_mode(AgentMode::Api, None);
        }
    }

    /// Use `model` for the next API chat turn (dropdown pick or a hand-typed id).
    pub fn select_api_model(&mut self, model: impl Into<String>) {
        let model = model.into();
        let model = model.trim().to_string();
        if !model.is_empty() {
            self.model = model;
        }
    }

    /// Merge models reported by the provider with the built-in candidates: the provider's order
    /// wins, candidates fill the gaps, and duplicates are dropped. Keeps a hand-typed model that
    /// the provider did not list.
    fn merge_api_models(&mut self, listed: Vec<String>) {
        let merged = merge_api_models(listed, self.provider.candidate_models());
        if merged.is_empty() {
            return;
        }
        self.api_models = merged;
        if self.model.is_empty() {
            self.model = self.api_models.first().cloned().unwrap_or_default();
        }
    }

    /// Switch CLI Agent and reset CLI model list.
    pub fn set_cli_agent(&mut self, c: CliAgent) {
        if self.cli_agent != c {
            self.cli_agent = c;
            if c != CliAgent::Custom {
                self.custom_cli_cmd = c.cmd_name().to_string();
            }
            self.cli_models = c.default_models();
            self.cli_model = self.cli_models.first().map(|m| m.id.clone()).unwrap_or_default();
            self.cli_scanning_models = false;
            self.disconnect_mode(AgentMode::Terminal, None);
        }
    }

    /// Status for a given mode.
    pub fn status_of(&self, mode: AgentMode) -> &ConnStatus {
        match mode {
            AgentMode::Api => &self.api_status,
            AgentMode::Terminal => &self.cli_status,
            AgentMode::Ollama => &self.ollama_status,
        }
    }

    /// Whether at least one backend (API, Terminal, or Ollama) is currently connected.
    pub fn any_connected(&self) -> bool {
        self.api_status.is_connected() || self.cli_status.is_connected() || self.ollama_status.is_connected()
    }

    /// Ensure `active_chat_backend` points to a connected backend if one exists.
    pub fn sync_active_backend(&mut self) {
        if self.status_of(self.active_chat_backend).is_connected() {
            return;
        }
        for m in [self.mode, AgentMode::Api, AgentMode::Terminal, AgentMode::Ollama] {
            if self.status_of(m).is_connected() {
                self.active_chat_backend = m;
                return;
            }
        }
    }

    /// Disconnect a single mode without disturbing the other modes.
    pub fn disconnect_mode(&mut self, target: AgentMode, reason: Option<String>) {
        let new_status = match reason {
            Some(msg) => ConnStatus::Failed(msg),
            None => ConnStatus::Disconnected,
        };
        match target {
            AgentMode::Api => {
                self.api_verify_cancel.store(true, Ordering::SeqCst);
                self.api_verify_rx = None;
                self.api_models_cancel.store(true, Ordering::SeqCst);
                self.api_models_rx = None;
                self.api_models_scanning = false;
                self.api_status = new_status;
            }
            AgentMode::Terminal => {
                self.cli_verify_cancel.store(true, Ordering::SeqCst);
                self.cli_verify_rx = None;
                self.cli_scanning_models = false;
                self.cli_status = new_status;
            }
            AgentMode::Ollama => {
                self.ollama_verify_cancel.store(true, Ordering::SeqCst);
                self.ollama_verify_rx = None;
                self.ollama_scanning = false;
                self.ollama_status = new_status;
            }
        }
        if self.active_chat_backend == target && self.running {
            self.cancel_task();
        }
        self.sync_active_backend();
    }

    /// Record an error notification in English and clear any broken/failed chat history.
    pub fn notify_error(&mut self, err: impl Into<String>) {
        let raw = err.into();
        let clean = format_english_error(&raw);
        self.error_notification = Some(clean.clone());
        self.pending_toast = Some(format!("Process Error: {clean}"));
    }

    /// Stop any active chat task immediately.
    pub fn cancel_task(&mut self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        if self.running {
            self.running = false;
            self.messages.clear();
            self.notify_error("Task was cancelled by user.");
        }
    }

    /// Stop active tasks and disconnect all backends when the AI Agent panel itself is closed.
    pub fn stop_and_disconnect(&mut self, reason: Option<String>) {
        self.cancel_flag.store(true, Ordering::SeqCst);
        self.api_verify_cancel.store(true, Ordering::SeqCst);
        self.cli_verify_cancel.store(true, Ordering::SeqCst);
        self.ollama_verify_cancel.store(true, Ordering::SeqCst);
        self.running = false;
        self.cli_scanning_models = false;
        self.ollama_scanning = false;
        self.worker_rx = None;
        self.api_verify_rx = None;
        self.cli_verify_rx = None;
        self.ollama_verify_rx = None;
        let st = match reason {
            Some(r) => ConnStatus::Failed(format_english_error(&r)),
            None => ConnStatus::Disconnected,
        };
        self.api_status = st.clone();
        self.cli_status = st.clone();
        self.ollama_status = st;
    }

    /// Poll background verification & chat worker events each frame.
    pub fn poll(&mut self, ctx: &egui::Context) {
        let mut events = Vec::new();
        for rx in [&self.api_verify_rx, &self.api_models_rx, &self.cli_verify_rx, &self.ollama_verify_rx, &self.worker_rx].into_iter().flatten() {
            while let Ok(ev) = rx.try_recv() {
                events.push(ev);
            }
        }
        if events.is_empty() {
            return;
        }
        for ev in events {
            match ev {
                WorkerEvent::ConnResult { mode, result, resolved_model, models } => {
                    if mode == AgentMode::Api
                        && let Some(m) = resolved_model
                        && !m.is_empty()
                    {
                        self.model = m;
                    }
                    if mode == AgentMode::Api
                        && let Some(list) = models.clone()
                    {
                        // The verification response also carries the account's model list.
                        self.merge_api_models(list);
                    }
                    if mode == AgentMode::Ollama
                        && let Some(list) = models
                    {
                        self.ollama_scanned = true;
                        self.ollama_scanning = false;
                        self.ollama_models = list;
                        if !self.ollama_models.is_empty() && !self.ollama_models.contains(&self.ollama_model) {
                            self.ollama_model = self.ollama_models[0].clone();
                        } else if self.ollama_models.is_empty() {
                            self.ollama_model.clear();
                        }
                    }
                    match result {
                        Ok(info) => {
                            let status = ConnStatus::Connected(info);
                            match mode {
                                AgentMode::Api => self.api_status = status,
                                AgentMode::Terminal => self.cli_status = status,
                                AgentMode::Ollama => self.ollama_status = status,
                            }
                            self.active_chat_backend = mode;
                            self.error_notification = None;
                        }
                        Err(err) => {
                            let eng_err = format_english_error(&err);
                            let status = ConnStatus::Failed(eng_err.clone());
                            match mode {
                                AgentMode::Api => self.api_status = status,
                                AgentMode::Terminal => self.cli_status = status,
                                AgentMode::Ollama => self.ollama_status = status,
                            }
                            self.messages.clear();
                            self.notify_error(format!("Failed to connect to {}: {eng_err}", mode.short_label()));
                        }
                    }
                }
                WorkerEvent::ApiModelsListed { models, error } => {
                    self.api_models_scanning = false;
                    self.api_models_error = error.map(|e| format_english_error(&e));
                    if !models.is_empty() {
                        self.merge_api_models(models);
                    }
                }
                WorkerEvent::OllamaModelsScanned { models, error } => {
                    self.ollama_scanned = true;
                    self.ollama_scanning = false;
                    self.ollama_scan_error = error.map(|e| format_english_error(&e));
                    self.ollama_models = models;
                    if !self.ollama_models.is_empty() {
                        if !self.ollama_models.contains(&self.ollama_model) {
                            self.ollama_model = self.ollama_models[0].clone();
                        }
                    } else {
                        self.ollama_model.clear();
                    }
                }
                WorkerEvent::CliModelsDiscovered { agent, models } => {
                    if self.cli_agent == agent && !models.is_empty() {
                        let mut merged = Vec::new();
                        for (id, display_name) in models {
                            let prev_access =
                                self.cli_models.iter().find(|m| m.id == id).map(|m| m.access.clone()).unwrap_or(CliModelAccess::Unknown);
                            merged.push(CliModelInfo { id, display_name, access: prev_access });
                        }
                        self.cli_models = merged;
                        if !self.cli_models.iter().any(|m| m.id == self.cli_model)
                            && let Some(first) = self.cli_models.first()
                        {
                            self.cli_model = first.id.clone();
                        }
                    }
                }
                WorkerEvent::CliModelVerified { agent, model_id, usable } => {
                    if self.cli_agent == agent {
                        let new_access = match usable {
                            Ok(()) => CliModelAccess::Usable,
                            Err(reason) => CliModelAccess::Unavailable(format_english_error(&reason)),
                        };
                        if let Some(item) = self.cli_models.iter_mut().find(|m| m.id == model_id) {
                            item.access = new_access;
                        }
                    }
                }
                WorkerEvent::CliScanFinished => {
                    self.cli_scanning_models = false;
                    // If current selected model was marked Unavailable and there is a Usable model, auto-select the Usable one if not yet connected
                    let cur_unavailable =
                        self.cli_models.iter().find(|m| m.id == self.cli_model).is_some_and(|m| matches!(m.access, CliModelAccess::Unavailable(_)));
                    if cur_unavailable && let Some(good) = self.cli_models.iter().find(|m| m.access == CliModelAccess::Usable) {
                        self.cli_model = good.id.clone();
                    }
                }
                WorkerEvent::ToolStart { name, args } => {
                    self.messages.push(ChatEntry::ToolCall {
                        name: name.clone(),
                        summary: if args.is_empty() { format!("Running {name}…") } else { args },
                        is_error: false,
                    });
                }
                WorkerEvent::ToolDone { name, summary, is_error } => {
                    if is_error {
                        self.notify_error(summary.clone());
                    }
                    if let Some(ChatEntry::ToolCall { name: n, summary: s, is_error: err }) = self.messages.last_mut()
                        && *n == name
                    {
                        *s = summary;
                        *err = is_error;
                    } else {
                        self.messages.push(ChatEntry::ToolCall { name, summary, is_error });
                    }
                }
                WorkerEvent::AssistantReply(text) => {
                    let clean = sanitize_and_compact_reply(&text);
                    if !clean.is_empty() {
                        self.messages.push(ChatEntry::Assistant(clean));
                    }
                }
                WorkerEvent::Error(e) => {
                    // When the process fails, clear the chat history and show an English error notification!
                    self.running = false;
                    self.messages.clear();
                    self.notify_error(e);
                }
                WorkerEvent::ConnectionLost { mode, reason } => {
                    // When connection drops mid-process, clear the chat history and show an English error notification!
                    self.running = false;
                    self.messages.clear();
                    let eng_reason = format_english_error(&reason);
                    self.notify_error(format!("{eng_reason} — active task stopped and closed."));
                    self.disconnect_mode(mode, Some(eng_reason));
                }
                WorkerEvent::Finished => {
                    self.running = false;
                }
            }
        }
        ctx.request_repaint();
    }
}

/// Normalize any error message into clean, concise English without markdown `*` characters.
fn format_english_error(raw: &str) -> String {
    let clean = raw.replace('*', "").trim().to_string();
    if clean.is_empty() {
        return "An unexpected error occurred during processing.".to_string();
    }
    let lower = clean.to_ascii_lowercase();
    if lower.contains("tidak ada koneksi internet") || lower.contains("koneksi internet terputus") {
        return "No internet connection detected. Please check your network and try again.".to_string();
    }
    if lower.contains("waktu tunggu habis") || lower.contains("timed out") || lower.contains("print timeout") {
        return "Request timed out while waiting for the AI model response. Please try again or switch models.".to_string();
    }
    if lower.contains("individual quota reached") || lower.contains("resource_exhausted") || lower.contains("insufficient_quota") {
        return "Model quota or rate limit reached for this account. Please select another model.".to_string();
    }
    if lower.contains("tidak ada tabel atau data") {
        return "No table or data found on the active sheet to process.".to_string();
    }
    if lower.contains("tidak memiliki kolom angka") {
        return "The active table has no numeric columns available for chart visualization.".to_string();
    }
    clean
}

/// Render the AI Agent pane inside the right-hand task pane.
pub fn show(app: &mut SheetApp, ui: &mut Ui, ctx: &egui::Context) {
    let t = Tokens::get(ctx);
    // The pane is right-anchored: keep every row inside the available width so long text can never
    // widen the panel (which used to slide its left edge over the grid).
    ui.set_width(ui.available_width());

    // Mode tabs. Switching tabs never disconnects a running backend.
    ui.horizontal_wrapped(|ui| {
        for (mode, label) in [(AgentMode::Api, "LLM API Key"), (AgentMode::Terminal, "Terminal Agent"), (AgentMode::Ollama, "Local (Ollama)")] {
            let selected = app.ai_agent.mode == mode;
            let connected = app.ai_agent.status_of(mode).is_connected();
            let text = if connected {
                RichText::new(format!("{label} (active)")).color(Color32::from_rgb(31, 157, 85)).strong()
            } else {
                RichText::new(label)
            };
            if ui.selectable_label(selected, text).clicked() {
                app.ai_agent.mode = mode;
                if connected {
                    app.ai_agent.active_chat_backend = mode;
                }
                if mode == AgentMode::Ollama && !app.ai_agent.ollama_scanned && !app.ai_agent.ollama_scanning {
                    start_ollama_scan(&mut app.ai_agent, ctx.clone());
                }
            }
        }
    });
    ui.add_space(6.0);

    // Auto-scan Ollama models the first time the Local tab is viewed.
    if app.ai_agent.mode == AgentMode::Ollama && !app.ai_agent.ollama_scanned && !app.ai_agent.ollama_scanning {
        start_ollama_scan(&mut app.ai_agent, ctx.clone());
    }

    let viewed_mode = app.ai_agent.mode;
    egui::Frame::NONE.fill(t.menu_bg).corner_radius(6.0).inner_margin(8.0).stroke(Stroke::new(1.0, t.separator)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        match viewed_mode {
            AgentMode::Api => show_api_config(app, ui, ctx, &t),
            AgentMode::Terminal => show_terminal_config(app, ui, ctx, &t),
            AgentMode::Ollama => show_ollama_config(app, ui, ctx, &t),
        }
    });

    ui.add_space(6.0);
    show_status_badge(app.ai_agent.status_of(viewed_mode), ui, &t);

    if let Some(err_msg) = app.ai_agent.error_notification.clone() {
        ui.add_space(6.0);
        let mut dismiss = false;
        let err_border = Color32::from_rgb(210, 50, 50);
        let err_bg = Color32::from_rgba_unmultiplied(210, 50, 50, 26);
        egui::Frame::NONE.fill(err_bg).corner_radius(6.0).inner_margin(8.0).stroke(Stroke::new(1.0, err_border)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new("Error").font(theme::ui_bold(12.5)).color(err_border));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("x").on_hover_text("Dismiss").clicked() {
                        dismiss = true;
                    }
                });
            });
            ui.add(egui::Label::new(RichText::new(err_msg).font(theme::ui_font(12.0)).color(t.text)).wrap());
        });
        if dismiss {
            app.ai_agent.error_notification = None;
        }
    }

    ui.add_space(6.0);
    ui.separator();
    show_history(app, ui, ctx, &t);
}

/// Cloud LLM API configuration: provider, API key, and the model to use.
fn show_api_config(app: &mut SheetApp, ui: &mut Ui, ctx: &egui::Context, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.label("Provider:");
        let current = app.ai_agent.provider;
        craft_ui::ComboBox::from_id_salt("ai_provider_combo")
            .selected_text(current.label())
            .width((ui.available_width() - 8.0).max(120.0))
            .show_ui(ui, |ui| {
                for &provider in ApiProvider::ALL {
                    if ui.selectable_label(provider == current, provider.label()).clicked() {
                        app.ai_agent.set_provider(provider);
                    }
                }
            })
            .response
            .accessible_name("Provider");
    });

    if app.ai_agent.provider == ApiProvider::Custom {
        ui.horizontal(|ui| {
            ui.label("Endpoint:");
            ui.add(
                egui::TextEdit::singleline(&mut app.ai_agent.base_url).hint_text("https://your-endpoint.com/v1").desired_width(ui.available_width()),
            );
        });
    }

    ui.horizontal(|ui| {
        ui.label("API Key:");
        let show_key = app.ai_agent.show_api_key;
        let toggle_label = if show_key { "Hide" } else { "Show" };
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button(toggle_label).clicked() {
                app.ai_agent.show_api_key = !app.ai_agent.show_api_key;
            }
            let response = ui.add(
                egui::TextEdit::singleline(&mut app.ai_agent.api_key)
                    .password(!show_key)
                    .hint_text("Enter API key (sk-...)")
                    .desired_width(ui.available_width()),
            );
            if response.changed() && matches!(app.ai_agent.api_status, ConnStatus::Failed(_)) {
                app.ai_agent.api_status = ConnStatus::Disconnected;
            }
        });
    });

    ui.horizontal(|ui| {
        ui.label("Model:");
        let selected = if app.ai_agent.model.is_empty() { "(choose a model)".to_string() } else { app.ai_agent.model.clone() };
        craft_ui::ComboBox::from_id_salt("ai_api_model_combo")
            .selected_text(selected)
            .width((ui.available_width() - 92.0).max(130.0))
            .show_ui(ui, |ui| {
                let models = app.ai_agent.api_models.clone();
                if models.is_empty() {
                    ui.label(RichText::new("No models loaded yet - refresh or type an id below.").small().color(t.text_dim));
                }
                for model in models {
                    let is_selected = app.ai_agent.model == model;
                    if ui.selectable_label(is_selected, model.as_str()).clicked() {
                        app.ai_agent.select_api_model(model);
                    }
                }
            })
            .response
            .accessible_name("Model");
        let scanning = app.ai_agent.api_models_scanning;
        let label = if scanning { "Loading..." } else { "Refresh" };
        if ui.add_enabled(!scanning, egui::Button::new(label)).on_hover_text("Load the models this API key can use (GET /models)").clicked() {
            start_api_model_scan(&mut app.ai_agent, ctx.clone());
        }
    });

    ui.horizontal(|ui| {
        ui.label("Custom model:");
        ui.add(
            egui::TextEdit::singleline(&mut app.ai_agent.model)
                .hint_text("e.g. deepseek-chat / deepseek-reasoner / gpt-5.5")
                .desired_width(ui.available_width()),
        );
    });

    if let Some(err) = app.ai_agent.api_models_error.clone() {
        ui.add(egui::Label::new(RichText::new(err).small().color(Color32::from_rgb(210, 50, 50))).wrap());
    } else if app.ai_agent.api_models.is_empty() {
        ui.add(
            egui::Label::new(RichText::new("Choose a model from the list or type an id, then connect.").small().italics().color(t.text_dim)).wrap(),
        );
    } else {
        ui.add(
            egui::Label::new(
                RichText::new(format!(
                    "{} models available (provider list + built-in candidates). Pick one or type another id; Connect verifies it first.",
                    app.ai_agent.api_models.len()
                ))
                .small()
                .color(t.text_dim),
            )
            .wrap(),
        );
    }

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let is_connecting = app.ai_agent.api_status.is_connecting();
        let is_connected = app.ai_agent.api_status.is_connected();
        let button_text = if is_connecting {
            "Connecting..."
        } else if is_connected {
            "Verify / Reconnect"
        } else {
            "Connect"
        };
        if ui.add_enabled(!is_connecting, egui::Button::new(button_text)).clicked() {
            start_api_verify(&mut app.ai_agent, ctx.clone());
        }
        if (is_connected || is_connecting) && ui.button("Disconnect").clicked() {
            app.ai_agent.disconnect_mode(AgentMode::Api, None);
        }
    });
}

/// Terminal CLI agent configuration: CLI agent, model and per-account model check.
fn show_terminal_config(app: &mut SheetApp, ui: &mut Ui, ctx: &egui::Context, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.label("CLI agent:");
        let current = app.ai_agent.cli_agent;
        craft_ui::ComboBox::from_id_salt("ai_cli_combo")
            .selected_text(current.label())
            .width((ui.available_width() - 8.0).max(120.0))
            .show_ui(ui, |ui| {
                for &agent in CliAgent::ALL {
                    if ui.selectable_label(agent == current, agent.label()).clicked() && agent != current {
                        app.ai_agent.set_cli_agent(agent);
                    }
                }
            })
            .response
            .accessible_name("CLI agent");
    });

    if app.ai_agent.cli_agent == CliAgent::Custom {
        ui.horizontal(|ui| {
            ui.label("Command:");
            ui.add(
                egui::TextEdit::singleline(&mut app.ai_agent.custom_cli_cmd).hint_text("e.g. agy, claude, codex").desired_width(ui.available_width()),
            );
        });
    }

    let selected_badge = app
        .ai_agent
        .cli_models
        .iter()
        .find(|model| model.id == app.ai_agent.cli_model)
        .map(CliModelInfo::badge_label)
        .unwrap_or_else(|| app.ai_agent.cli_model.clone());

    let mut model_changed_to: Option<String> = None;
    ui.horizontal(|ui| {
        ui.label("Model:");
        craft_ui::ComboBox::from_id_salt("ai_cli_model_combo")
            .selected_text(selected_badge)
            .width((ui.available_width() - 78.0).max(110.0))
            .show_ui(ui, |ui| {
                let models = app.ai_agent.cli_models.clone();
                for item in models {
                    let is_selected = app.ai_agent.cli_model == item.id;
                    let label = match &item.access {
                        CliModelAccess::Usable => RichText::new(item.badge_label()).color(Color32::from_rgb(31, 157, 85)),
                        CliModelAccess::Unavailable(_) => RichText::new(item.badge_label()).color(Color32::from_rgb(210, 50, 50)),
                        CliModelAccess::Unknown => RichText::new(item.badge_label()),
                    };
                    let response = ui.selectable_label(is_selected, label);
                    let response = if let CliModelAccess::Unavailable(reason) = &item.access {
                        response.on_hover_text(format!("Unavailable: {reason}"))
                    } else {
                        response
                    };
                    if response.clicked() && !is_selected {
                        model_changed_to = Some(item.id.clone());
                    }
                }
            })
            .response
            .accessible_name("Model");

        let scanning = app.ai_agent.cli_scanning_models || app.ai_agent.cli_status.is_connecting();
        let scan_label = if app.ai_agent.cli_scanning_models { "Checking..." } else { "Check" };
        if ui.add_enabled(!scanning, egui::Button::new(scan_label)).on_hover_text("Check which models your account can use").clicked() {
            start_cli_scan_all_models(&mut app.ai_agent, ctx.clone());
        }
    });

    if let Some(new_model) = model_changed_to {
        app.ai_agent.cli_model = new_model;
        start_cli_verify(&mut app.ai_agent, ctx.clone());
    }

    let usable_count = app.ai_agent.cli_models.iter().filter(|model| model.access == CliModelAccess::Usable).count();
    let unavailable_count = app.ai_agent.cli_models.iter().filter(|model| matches!(model.access, CliModelAccess::Unavailable(_))).count();
    if usable_count > 0 || unavailable_count > 0 {
        ui.label(RichText::new(format!("Account models: {usable_count} usable, {unavailable_count} unavailable")).small().color(t.text_dim));
    } else {
        ui.add(egui::Label::new(RichText::new("Pick a model, then connect to verify it.").small().italics().color(t.text_dim)).wrap());
    }

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let is_connecting = app.ai_agent.cli_status.is_connecting();
        let is_connected = app.ai_agent.cli_status.is_connected();
        let button_text = if is_connecting {
            "Verifying..."
        } else if is_connected {
            "Verify / Reconnect"
        } else {
            "Connect"
        };
        if ui.add_enabled(!is_connecting, egui::Button::new(button_text)).clicked() {
            start_cli_verify(&mut app.ai_agent, ctx.clone());
        }
        if (is_connected || is_connecting) && ui.button("Disconnect").clicked() {
            app.ai_agent.disconnect_mode(AgentMode::Terminal, None);
        }
    });
}

/// Local Ollama configuration: URL and installed models.
fn show_ollama_config(app: &mut SheetApp, ui: &mut Ui, ctx: &egui::Context, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.label("Ollama URL:");
        ui.add(egui::TextEdit::singleline(&mut app.ai_agent.ollama_url).hint_text("http://127.0.0.1:11434").desired_width(ui.available_width()));
    });

    ui.horizontal(|ui| {
        ui.label("Installed model:");
        let has_models = !app.ai_agent.ollama_models.is_empty();
        let selected_text = if app.ai_agent.ollama_scanning {
            "Scanning Ollama...".to_string()
        } else if has_models {
            app.ai_agent.ollama_model.clone()
        } else {
            "(no models installed)".to_string()
        };

        ui.add_enabled_ui(has_models && !app.ai_agent.ollama_scanning, |ui| {
            craft_ui::ComboBox::from_id_salt("ai_ollama_model_combo")
                .selected_text(selected_text)
                .width((ui.available_width() - 78.0).max(110.0))
                .show_ui(ui, |ui| {
                    let models = app.ai_agent.ollama_models.clone();
                    for model in models {
                        if ui.selectable_label(app.ai_agent.ollama_model == model, &model).clicked() {
                            app.ai_agent.ollama_model = model;
                        }
                    }
                })
                .response
                .accessible_name("Installed model");
        });

        let scan_label = if app.ai_agent.ollama_scanning { "Scanning..." } else { "Scan" };
        if ui.add_enabled(!app.ai_agent.ollama_scanning, egui::Button::new(scan_label)).on_hover_text("Scan the models installed in Ollama").clicked()
        {
            start_ollama_scan(&mut app.ai_agent, ctx.clone());
        }
    });

    if app.ai_agent.ollama_scanning {
        ui.label(RichText::new("Detecting models installed in Ollama...").small().italics().color(t.text_dim));
    } else if let Some(err) = &app.ai_agent.ollama_scan_error {
        ui.add(egui::Label::new(RichText::new(err.clone()).small().color(Color32::from_rgb(210, 50, 50))).wrap());
    } else if app.ai_agent.ollama_scanned && app.ai_agent.ollama_models.is_empty() {
        ui.add(
            egui::Label::new(
                RichText::new("No models installed. Run `ollama pull llama3.2` in a terminal, then Scan.")
                    .small()
                    .color(Color32::from_rgb(220, 140, 0)),
            )
            .wrap(),
        );
    } else if !app.ai_agent.ollama_models.is_empty() {
        ui.label(
            RichText::new(format!("{} models installed in Ollama.", app.ai_agent.ollama_models.len())).small().color(Color32::from_rgb(31, 157, 85)),
        );
    }

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let is_connecting = app.ai_agent.ollama_status.is_connecting();
        let is_connected = app.ai_agent.ollama_status.is_connected();
        let can_connect = !is_connecting && !app.ai_agent.ollama_models.is_empty();
        let button_text = if is_connecting {
            "Connecting..."
        } else if is_connected {
            "Reconnect"
        } else {
            "Connect"
        };
        if ui.add_enabled(can_connect, egui::Button::new(button_text)).clicked() {
            start_ollama_verify(&mut app.ai_agent, ctx.clone());
        }
        if (is_connected || is_connecting) && ui.button("Disconnect").clicked() {
            app.ai_agent.disconnect_mode(AgentMode::Ollama, None);
        }
    });
}

fn show_status_badge(status: &ConnStatus, ui: &mut Ui, t: &Tokens) {
    let (dot_color, text) = match status {
        ConnStatus::Disconnected => (t.text_dim, "Status: not connected".to_string()),
        ConnStatus::Connecting => (Color32::from_rgb(230, 168, 0), "Status: verifying connection and model...".to_string()),
        ConnStatus::Connected(info) => (Color32::from_rgb(31, 157, 85), format!("Connected: {info}")),
        ConnStatus::Failed(err) => (Color32::from_rgb(210, 50, 50), format!("Connection failed: {err}")),
    };
    ui.horizontal_wrapped(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.5, dot_color);
        ui.add(egui::Label::new(RichText::new(text).color(dot_color).strong()).wrap());
    });
}

/// The pane's single subsection: what the agent has been asked and what it did.
fn show_history(app: &mut SheetApp, ui: &mut Ui, ctx: &egui::Context, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("History").font(theme::ui_bold(13.0)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if app.ai_agent.running && ui.small_button("Stop").clicked() {
                app.ai_agent.cancel_task();
            }
            if !app.ai_agent.messages.is_empty() && ui.small_button("Clear").on_hover_text("Clear history").clicked() {
                app.ai_agent.messages.clear();
            }
        });
    });
    ui.add_space(4.0);

    let connected = app.ai_agent.any_connected();
    let input_height = if connected { 86.0 } else { 0.0 };
    let history_height = (ui.available_height() - input_height - 12.0).max(120.0);

    if app.ai_agent.messages.is_empty() {
        ui.add(
            egui::Label::new(
                RichText::new("No activity yet. Ask the agent to do something and its actions will appear here.").small().italics().color(t.text_dim),
            )
            .wrap(),
        );
        ui.add_space(4.0);
    } else {
        egui::ScrollArea::vertical().id_salt("ai_history_scroll").max_height(history_height).auto_shrink([false, false]).stick_to_bottom(true).show(
            ui,
            |ui| {
                ui.set_width(ui.available_width());
                for entry in &app.ai_agent.messages {
                    match entry {
                        ChatEntry::User(message) => {
                            egui::Frame::NONE
                                .fill(t.accent.gamma_multiply(0.18))
                                .corner_radius(6.0)
                                .inner_margin(7.0)
                                .stroke(Stroke::new(1.0, t.accent.gamma_multiply(0.5)))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(RichText::new("You").small().strong().color(t.accent));
                                    ui.add(egui::Label::new(message).wrap());
                                });
                        }
                        ChatEntry::Assistant(message) => {
                            egui::Frame::NONE.fill(t.menu_bg).corner_radius(6.0).inner_margin(8.0).stroke(Stroke::new(1.0, t.separator)).show(
                                ui,
                                |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(RichText::new("AI Agent").small().strong().color(Color32::from_rgb(31, 157, 85)));
                                    ui.add_space(2.0);
                                    render_clean_assistant_message(ui, message, t);
                                },
                            );
                        }
                        ChatEntry::ToolCall { name: _, summary, is_error } => {
                            let accent_color = if *is_error { Color32::from_rgb(210, 50, 50) } else { Color32::from_rgb(31, 157, 85) };
                            let icon = if *is_error { "X" } else { "OK" };
                            let clean_summary = summary.replace('*', "");
                            egui::Frame::NONE
                                .fill(t.window)
                                .corner_radius(4.0)
                                .inner_margin(egui::Margin::symmetric(7, 4))
                                .stroke(Stroke::new(1.0, accent_color.gamma_multiply(0.45)))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(RichText::new(icon).small().strong().color(accent_color));
                                        ui.add(
                                            egui::Label::new(RichText::new(clean_summary).small().color(if *is_error {
                                                accent_color
                                            } else {
                                                t.text
                                            }))
                                            .wrap(),
                                        );
                                    });
                                });
                        }
                        ChatEntry::System(message) => {
                            ui.add(egui::Label::new(RichText::new(message).small().italics().color(t.text_dim)).wrap());
                        }
                    }
                    ui.add_space(4.0);
                }
                if app.ai_agent.running {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new("Agent is working...").small().italics().color(t.text_dim));
                    });
                }
            },
        );
    }

    ui.add_space(4.0);

    if !connected {
        ui.add(egui::Label::new(RichText::new("Connect a backend above to give the agent work.").small().italics().color(t.text_dim)).wrap());
        return;
    }

    let mut trigger_send = false;
    ui.horizontal(|ui| {
        let response = ui.add_sized(
            vec2((ui.available_width() - 62.0).max(80.0), 46.0),
            egui::TextEdit::multiline(&mut app.ai_agent.chat_input)
                .hint_text("Ask the AI Agent to build data, formulas, tables, formats, charts... (Enter to send)"),
        );
        if response.has_focus()
            && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift)
            && !app.ai_agent.chat_input.trim().is_empty()
            && !app.ai_agent.running
        {
            trigger_send = true;
        }
        ui.vertical(|ui| {
            let can_send = !app.ai_agent.running && !app.ai_agent.chat_input.trim().is_empty();
            if ui.add_enabled(can_send, egui::Button::new("Send")).clicked() {
                trigger_send = true;
            }
            if app.ai_agent.running && ui.small_button("Stop").clicked() {
                app.ai_agent.cancel_task();
            }
        });
    });

    if trigger_send {
        let prompt = app.ai_agent.chat_input.trim().to_string();
        app.ai_agent.chat_input.clear();
        send_chat_prompt(&mut app.ai_agent, prompt, ctx.clone());
    }
}

/// Sanitize and compact AI replies so there are ZERO raw `*` / markdown symbols and the output is concise & tidy.
fn sanitize_and_compact_reply(raw: &str) -> String {
    let mut cleaned_lines: Vec<String> = Vec::new();
    let mut in_code_fence = false;

    for raw_line in raw.lines() {
        let trimmed = raw_line.trim();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }
        if trimmed.is_empty() {
            if cleaned_lines.last().is_some_and(|l| !l.is_empty()) {
                cleaned_lines.push(String::new());
            }
            continue;
        }
        // Skip markdown horizontal dividers (`---`, `***`, `___`)
        if trimmed.chars().all(|c| matches!(c, '-' | '*' | '_' | ' ')) && trimmed.len() >= 3 {
            continue;
        }

        // Detect bullet lines (`* item`, `- item`, `+ item`, `• item`) before stripping `*`
        let is_bullet = trimmed.starts_with("* ") || trimmed.starts_with("- ") || trimmed.starts_with("+ ") || trimmed.starts_with("• ");
        let body = if is_bullet { trimmed.trim_start_matches(['*', '-', '+', '•']).trim_start() } else { trimmed };

        // Strip all markdown `*`, `` ` ``, and leading `#` characters
        let no_hashes = body.trim_start_matches('#').trim_start();
        let no_stars: String = no_hashes.chars().filter(|&c| c != '*' && c != '`').collect();
        let final_line = no_stars.trim();
        if final_line.is_empty() {
            continue;
        }

        if is_bullet {
            cleaned_lines.push(format!("• {final_line}"));
        } else {
            cleaned_lines.push(final_line.to_string());
        }
    }

    while cleaned_lines.last().is_some_and(String::is_empty) {
        cleaned_lines.pop();
    }

    // Keep the output concise (max 10 non-empty lines) so the user is never overwhelmed by a wall of text
    let mut out_lines = Vec::new();
    let mut non_empty_count = 0usize;
    for l in cleaned_lines {
        if l.is_empty() {
            if non_empty_count > 0 && non_empty_count < 10 {
                out_lines.push(l);
            }
        } else {
            non_empty_count += 1;
            if non_empty_count <= 10 {
                out_lines.push(l);
            }
        }
    }
    while out_lines.last().is_some_and(String::is_empty) {
        out_lines.pop();
    }

    let joined = out_lines.join("\n");
    if joined.trim().is_empty() { "Done.".to_string() } else { joined }
}

/// Render an assistant message with clean native typography, structured bullets, bold labels, and zero `*` characters.
fn render_clean_assistant_message(ui: &mut Ui, raw_msg: &str, t: &Tokens) {
    let clean = sanitize_and_compact_reply(raw_msg);
    for line in clean.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            ui.add_space(3.0);
            continue;
        }

        if let Some(bullet_body) = trimmed.strip_prefix("• ") {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("•").strong().color(t.accent));
                render_key_value_line(ui, bullet_body, t);
            });
        } else if let Some((num_prefix, rest)) = split_numbered_prefix(trimmed) {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(num_prefix).strong().color(t.accent));
                render_key_value_line(ui, rest, t);
            });
        } else if trimmed.ends_with(':') && trimmed.chars().count() <= 60 {
            ui.label(RichText::new(trimmed).font(theme::ui_bold(12.5)).color(t.text));
        } else {
            render_key_value_line(ui, trimmed, t);
        }
    }
}

fn split_numbered_prefix(s: &str) -> Option<(&str, &str)> {
    let dot_pos = s.find(". ")?;
    if dot_pos == 0 || dot_pos > 3 {
        return None;
    }
    let prefix = &s[..dot_pos];
    if prefix.chars().all(|c| c.is_ascii_digit()) { Some((&s[..=dot_pos], s[dot_pos + 2..].trim_start())) } else { None }
}

fn render_key_value_line(ui: &mut Ui, text: &str, t: &Tokens) {
    let clean_text = text.replace('*', "");
    if let Some((left, right)) = clean_text.split_once(':') {
        let left_trim = left.trim();
        let right_trim = right.trim();
        if !left_trim.is_empty() && !right_trim.is_empty() && left_trim.chars().count() <= 38 && !left_trim.contains("http") {
            let mut job = egui::text::LayoutJob::default();
            job.append(&format!("{left_trim}: "), 0.0, egui::TextFormat { font_id: theme::ui_bold(12.0), color: t.text, ..Default::default() });
            job.append(right_trim, 0.0, egui::TextFormat { font_id: theme::ui_font(12.0), color: t.text, ..Default::default() });
            ui.label(job);
            return;
        }
    }
    ui.label(RichText::new(clean_text).font(theme::ui_font(12.0)).color(t.text));
}

// ============================================================================
// Network & Endpoint Reachability Checks
// ============================================================================

/// Rewrite a local endpoint URL to the loopback IP. Pure string handling, so it also compiles for
/// the web build (the connection tabs are shown there too, even though they cannot connect).
fn normalize_local_url(url: &str) -> String {
    url.replace("://localhost", "://127.0.0.1")
}

#[cfg(not(target_arch = "wasm32"))]
fn check_endpoint_reachable(url: &str) -> bool {
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;

    let norm = normalize_local_url(url);
    if let Some(host_port) = extract_host_port(&norm)
        && let Ok(addrs) = host_port.to_socket_addrs()
    {
        for addr in addrs {
            if TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok() {
                return true;
            }
        }
    }
    false
}

#[cfg(not(target_arch = "wasm32"))]
fn check_internet_online(url_hint: Option<&str>) -> bool {
    use std::net::{SocketAddr, TcpStream};
    use std::time::Duration;

    if let Some(u) = url_hint
        && check_endpoint_reachable(u)
    {
        return true;
    }
    let fallbacks: &[SocketAddr] = &[SocketAddr::from(([1, 1, 1, 1], 443)), SocketAddr::from(([8, 8, 8, 8], 53))];
    for addr in fallbacks {
        if TcpStream::connect_timeout(addr, Duration::from_secs(2)).is_ok() {
            return true;
        }
    }
    false
}

#[cfg(target_arch = "wasm32")]
fn check_internet_online(_url_hint: Option<&str>) -> bool {
    true
}

#[cfg(not(target_arch = "wasm32"))]
fn extract_host_port(url: &str) -> Option<String> {
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
    let default_port = if url.starts_with("https://") { 443 } else { 80 };
    let host_part = rest.split('/').next()?;
    if host_part.contains(':') { Some(host_part.to_string()) } else { Some(format!("{host_part}:{default_port}")) }
}

// ============================================================================
// Independent Verification & Model Discovery Workers
// ============================================================================

fn start_api_verify(state: &mut AiAgentState, ctx: egui::Context) {
    let key = state.api_key.trim().to_string();
    if key.is_empty() {
        let msg = "API key cannot be empty.";
        state.api_status = ConnStatus::Failed(msg.into());
        state.notify_error(msg);
        return;
    }
    let provider = state.provider;
    let base_url = if provider == ApiProvider::Custom {
        state.base_url.trim().trim_end_matches('/').to_string()
    } else {
        provider.default_base_url().to_string()
    };
    let preferred = state.model.trim().to_string();

    state.api_verify_cancel.store(true, Ordering::SeqCst);
    let cancel = Arc::new(AtomicBool::new(false));
    state.api_verify_cancel = Arc::clone(&cancel);
    state.api_status = ConnStatus::Connecting;
    state.error_notification = None;

    let (tx, rx) = channel();
    state.api_verify_rx = Some(rx);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-verify-api".into()).spawn(move || {
            if !check_internet_online(Some(&base_url)) {
                let _ = tx.send(WorkerEvent::ConnResult {
                    mode: AgentMode::Api,
                    result: Err("No internet connection detected.".into()),
                    resolved_model: None,
                    models: None,
                });
                ctx.request_repaint();
                return;
            }
            let pick = if preferred.trim().is_empty() { None } else { Some(preferred.trim()) };
            let (res, resolved_model, models) = verify_api_connection_auto_model(provider, &base_url, &key, &cancel, pick);
            if !cancel.load(Ordering::SeqCst) {
                let _ = tx.send(WorkerEvent::ConnResult { mode: AgentMode::Api, result: res, resolved_model, models: Some(models) });
                ctx.request_repaint();
            }
        });
    }
}

/// Fetch the model list the API key can use (`GET {base_url}/models`) and merge it into the
/// dropdown, without connecting or changing the current model.
fn start_api_model_scan(state: &mut AiAgentState, ctx: egui::Context) {
    let provider = state.provider;
    let base_url = if provider == ApiProvider::Custom {
        state.base_url.trim().trim_end_matches('/').to_string()
    } else {
        provider.default_base_url().to_string()
    };
    let key = state.api_key.trim().to_string();
    if key.is_empty() {
        let msg = "Enter the API key first to load the provider's model list.";
        state.api_models_error = Some(msg.into());
        state.notify_error(msg);
        return;
    }

    state.api_models_cancel.store(true, Ordering::SeqCst);
    let cancel = Arc::new(AtomicBool::new(false));
    state.api_models_cancel = Arc::clone(&cancel);
    state.api_models_scanning = true;
    state.api_models_error = None;

    let (tx, rx) = channel();
    state.api_models_rx = Some(rx);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-scan-api-models".into()).spawn(move || {
            if !check_internet_online(Some(&base_url)) {
                let _ = tx.send(WorkerEvent::ApiModelsListed { models: Vec::new(), error: Some("No internet connection detected.".into()) });
                ctx.request_repaint();
                return;
            }
            match fetch_api_models_list(provider, &base_url, &key, &cancel) {
                Ok(models) => {
                    let _ = tx.send(WorkerEvent::ApiModelsListed { models, error: None });
                }
                Err(e) => {
                    let _ = tx.send(WorkerEvent::ApiModelsListed { models: Vec::new(), error: Some(e) });
                }
            }
            ctx.request_repaint();
        });
    }
}

fn start_cli_verify(state: &mut AiAgentState, ctx: egui::Context) {
    let cli = state.cli_agent;
    let cmd_str = if cli == CliAgent::Custom { state.custom_cli_cmd.trim().to_string() } else { cli.cmd_name().to_string() };
    if cmd_str.is_empty() {
        let msg = "CLI command cannot be empty.";
        state.cli_status = ConnStatus::Failed(msg.into());
        state.notify_error(msg);
        return;
    }
    let chosen_model = state.cli_model.trim().to_string();

    state.cli_verify_cancel.store(true, Ordering::SeqCst);
    let cancel = Arc::new(AtomicBool::new(false));
    state.cli_verify_cancel = Arc::clone(&cancel);
    state.cli_status = ConnStatus::Connecting;
    state.error_notification = None;

    let (tx, rx) = channel();
    state.cli_verify_rx = Some(rx);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-verify-cli".into()).spawn(move || {
            if !check_internet_online(None) {
                let _ = tx.send(WorkerEvent::ConnResult {
                    mode: AgentMode::Terminal,
                    result: Err("No internet connection detected.".into()),
                    resolved_model: None,
                    models: None,
                });
                ctx.request_repaint();
                return;
            }

            // 1. Discover dynamic models if supported (e.g. `agy models` or `~/.codex/models_cache.json`)
            if cli == CliAgent::Agy {
                ensure_clean_agy_mcp_config();
                if let Some(discovered) = discover_agy_models(&cmd_str, &cancel) {
                    let _ = tx.send(WorkerEvent::CliModelsDiscovered { agent: cli, models: discovered });
                    ctx.request_repaint();
                }
            } else if cli == CliAgent::Codex
                && let Some(discovered) = discover_codex_models_from_disk()
            {
                let _ = tx.send(WorkerEvent::CliModelsDiscovered { agent: cli, models: discovered });
                ctx.request_repaint();
            }

            if cancel.load(Ordering::SeqCst) {
                return;
            }

            // 2. Actively verify whether the user's account can use `chosen_model`!
            let verify_res = verify_cli_model_usable(cli, &cmd_str, &chosen_model, &cancel);
            if cancel.load(Ordering::SeqCst) {
                return;
            }

            match verify_res {
                Ok(info) => {
                    let _ = tx.send(WorkerEvent::CliModelVerified { agent: cli, model_id: chosen_model.clone(), usable: Ok(()) });
                    let _ = tx.send(WorkerEvent::ConnResult {
                        mode: AgentMode::Terminal,
                        result: Ok(info),
                        resolved_model: Some(chosen_model),
                        models: None,
                    });
                }
                Err(err_reason) => {
                    let _ = tx.send(WorkerEvent::CliModelVerified { agent: cli, model_id: chosen_model.clone(), usable: Err(err_reason.clone()) });
                    let _ = tx.send(WorkerEvent::ConnResult {
                        mode: AgentMode::Terminal,
                        result: Err(format!("Model '{chosen_model}' is not available for your account: {err_reason}")),
                        resolved_model: None,
                        models: None,
                    });
                }
            }
            ctx.request_repaint();
        });
    }
}

/// Probe representative models for the selected CLI agent so the dropdown shows OK / X for each model!
fn start_cli_scan_all_models(state: &mut AiAgentState, ctx: egui::Context) {
    let cli = state.cli_agent;
    let cmd_str = if cli == CliAgent::Custom { state.custom_cli_cmd.trim().to_string() } else { cli.cmd_name().to_string() };
    if cmd_str.is_empty() {
        return;
    }

    state.cli_verify_cancel.store(true, Ordering::SeqCst);
    let cancel = Arc::new(AtomicBool::new(false));
    state.cli_verify_cancel = Arc::clone(&cancel);
    state.cli_scanning_models = true;
    state.cli_status = ConnStatus::Connecting;
    state.error_notification = None;

    let selected_model = state.cli_model.clone();
    let models_to_check: Vec<String> = state.cli_models.iter().map(|m| m.id.clone()).collect();

    let (tx, rx) = channel();
    state.cli_verify_rx = Some(rx);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-scan-cli-models".into()).spawn(move || {
            if !check_internet_online(None) {
                let _ = tx.send(WorkerEvent::ConnResult {
                    mode: AgentMode::Terminal,
                    result: Err("No internet connection detected.".into()),
                    resolved_model: None,
                    models: None,
                });
                let _ = tx.send(WorkerEvent::CliScanFinished);
                ctx.request_repaint();
                return;
            }

            if cli == CliAgent::Agy {
                ensure_clean_agy_mcp_config();
                if let Some(discovered) = discover_agy_models(&cmd_str, &cancel) {
                    let _ = tx.send(WorkerEvent::CliModelsDiscovered { agent: cli, models: discovered });
                    ctx.request_repaint();
                }
            } else if cli == CliAgent::Codex
                && let Some(discovered) = discover_codex_models_from_disk()
            {
                let _ = tx.send(WorkerEvent::CliModelsDiscovered { agent: cli, models: discovered });
                ctx.request_repaint();
            }

            // Group models by family so we don't have to run 13 slow sequential probes when 4 family probes cover all tiers!
            let mut family_results: std::collections::HashMap<String, Result<(), String>> = std::collections::HashMap::new();
            let mut first_working: Option<(String, String)> = None;
            let mut selected_working: Option<String> = None;
            let mut last_err = String::from("No usable models found for your CLI account.");

            // Check the currently selected model first so connection happens fast
            let mut ordered = vec![selected_model.clone()];
            for m in models_to_check {
                if m != selected_model {
                    ordered.push(m);
                }
            }

            for m_id in ordered {
                if cancel.load(Ordering::SeqCst) {
                    return;
                }
                let fam = model_family_key(cli, &m_id);
                let res = if let Some(cached) = family_results.get(&fam) {
                    cached.clone()
                } else {
                    let r = verify_cli_model_usable(cli, &cmd_str, &m_id, &cancel).map(|info| {
                        if first_working.is_none() {
                            first_working = Some((m_id.clone(), info.clone()));
                        }
                        if m_id == selected_model {
                            selected_working = Some(info);
                        }
                    });
                    family_results.insert(fam, r.clone());
                    r
                };

                match &res {
                    Ok(()) => {
                        if first_working.is_none() {
                            first_working = Some((m_id.clone(), format!("{cmd_str} ({m_id}) [OK Verified]")));
                        }
                    }
                    Err(e) => {
                        last_err = e.clone();
                    }
                }

                let _ = tx.send(WorkerEvent::CliModelVerified { agent: cli, model_id: m_id, usable: res });
                ctx.request_repaint();
            }

            let _ = tx.send(WorkerEvent::CliScanFinished);
            if let Some(info) = selected_working {
                let _ = tx.send(WorkerEvent::ConnResult {
                    mode: AgentMode::Terminal,
                    result: Ok(info),
                    resolved_model: Some(selected_model),
                    models: None,
                });
            } else if let Some((best_model, info)) = first_working {
                let _ =
                    tx.send(WorkerEvent::ConnResult { mode: AgentMode::Terminal, result: Ok(info), resolved_model: Some(best_model), models: None });
            } else {
                let _ = tx.send(WorkerEvent::ConnResult { mode: AgentMode::Terminal, result: Err(last_err), resolved_model: None, models: None });
            }
            ctx.request_repaint();
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn model_family_key(cli: CliAgent, model_id: &str) -> String {
    if cli == CliAgent::Agy {
        for suffix in ["-low", "-medium", "-high"] {
            if let Some(stem) = model_id.strip_suffix(suffix) {
                return stem.to_string();
            }
        }
    }
    model_id.to_string()
}

/// Scan installed Ollama models without connecting yet (or when Refresh/Scan is clicked).
fn start_ollama_scan(state: &mut AiAgentState, ctx: egui::Context) {
    let base_url = normalize_local_url(state.ollama_url.trim().trim_end_matches('/'));
    state.ollama_scanning = true;
    state.ollama_scan_error = None;

    let cancel = Arc::new(AtomicBool::new(false));
    state.ollama_verify_cancel = Arc::clone(&cancel);

    let (tx, rx) = channel();
    state.ollama_verify_rx = Some(rx);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-scan-ollama".into()).spawn(move || {
            let (models, err) = discover_installed_ollama_models(&base_url, &cancel);
            if !cancel.load(Ordering::SeqCst) {
                let _ = tx.send(WorkerEvent::OllamaModelsScanned { models, error: err });
                ctx.request_repaint();
            }
        });
    }
}

fn start_ollama_verify(state: &mut AiAgentState, ctx: egui::Context) {
    let base_url = normalize_local_url(state.ollama_url.trim().trim_end_matches('/'));
    if base_url.is_empty() {
        let msg = "Ollama server URL cannot be empty.";
        state.ollama_status = ConnStatus::Failed(msg.into());
        state.notify_error(msg);
        return;
    }
    let chosen_model = state.ollama_model.trim().to_string();

    state.ollama_verify_cancel.store(true, Ordering::SeqCst);
    let cancel = Arc::new(AtomicBool::new(false));
    state.ollama_verify_cancel = Arc::clone(&cancel);
    state.ollama_status = ConnStatus::Connecting;
    state.error_notification = None;

    let (tx, rx) = channel();
    state.ollama_verify_rx = Some(rx);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-verify-ollama".into()).spawn(move || {
            let (models, err) = discover_installed_ollama_models(&base_url, &cancel);
            if cancel.load(Ordering::SeqCst) {
                return;
            }
            if let Some(e) = err
                && models.is_empty()
            {
                let _ = tx.send(WorkerEvent::ConnResult { mode: AgentMode::Ollama, result: Err(e), resolved_model: None, models: Some(Vec::new()) });
                ctx.request_repaint();
                return;
            }
            if models.is_empty() {
                let _ = tx.send(WorkerEvent::ConnResult {
                    mode: AgentMode::Ollama,
                    result: Err("No models are installed in Ollama. Run `ollama pull llama3.2` in your terminal first.".into()),
                    resolved_model: None,
                    models: Some(Vec::new()),
                });
                ctx.request_repaint();
                return;
            }
            let active_model = if models.iter().any(|m| m == &chosen_model) { chosen_model } else { models[0].clone() };
            let _ = tx.send(WorkerEvent::ConnResult {
                mode: AgentMode::Ollama,
                result: Ok(format!("Ollama ({active_model} — {} installed)", models.len())),
                resolved_model: Some(active_model),
                models: Some(models),
            });
            ctx.request_repaint();
        });
    }
}

/// Detect installed models in Ollama using both `GET /api/tags` and `ollama list` CLI.
#[cfg(not(target_arch = "wasm32"))]
fn discover_installed_ollama_models(base_url: &str, cancel: &AtomicBool) -> (Vec<String>, Option<String>) {
    let mut models = Vec::new();
    let mut api_err = None;

    if check_endpoint_reachable(base_url) {
        let tags_url = format!("{base_url}/api/tags");
        let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
        match run_process_cancellable_with_stdin(curl_bin, &["-sS", "--max-time", "6", &tags_url], None, None, cancel, true, 10) {
            Ok(out) => {
                if let Ok(parsed) = serde_json::from_str::<Value>(&out)
                    && let Some(arr) = parsed.get("models").and_then(Value::as_array)
                {
                    for m in arr {
                        if let Some(name) = m.get("name").or_else(|| m.get("model")).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
                            && !models.iter().any(|x| x == name)
                        {
                            models.push(name.to_string());
                        }
                    }
                }
            }
            Err(e) => api_err = Some(format!("Failed to access Ollama API: {e}")),
        }
    } else {
        api_err = Some(format!("Ollama server at {base_url} is not reachable (make sure Ollama is running)."));
    }

    // Also check `ollama list` CLI so installed models are detected even if URL differs
    if let Ok(cli_out) = run_process_cancellable_with_stdin("ollama", &["list"], None, None, cancel, true, 6) {
        for line in cli_out.lines().skip(1) {
            if let Some(first_col) = line.split_whitespace().next() {
                let name = first_col.trim();
                if !name.is_empty() && !name.eq_ignore_ascii_case("NAME") && !models.iter().any(|x| x == name) {
                    models.push(name.to_string());
                }
            }
        }
        if !models.is_empty() {
            api_err = None;
        }
    }

    (models, api_err)
}

/// Verify the API key and resolve the model to use. Tries the user's pick first, then the models
/// the provider reports for this key, then the built-in candidates. Also returns the model list the
/// UI shows in its dropdown.
#[cfg(not(target_arch = "wasm32"))]
fn verify_api_connection_auto_model(
    provider: ApiProvider,
    base_url: &str,
    api_key: &str,
    cancel: &AtomicBool,
    preferred: Option<&str>,
) -> (Result<String, String>, Option<String>, Vec<String>) {
    let discovered = fetch_api_models_list(provider, base_url, api_key, cancel).unwrap_or_default();
    let candidates = build_api_candidate_order(preferred, &discovered, provider.candidate_models());
    let models = merge_api_models(discovered, provider.candidate_models());

    let endpoint = if provider == ApiProvider::Anthropic { format!("{base_url}/messages") } else { format!("{base_url}/chat/completions") };

    let mut last_err = String::from("Failed to verify API key.");

    for candidate in &candidates {
        if cancel.load(Ordering::SeqCst) {
            return (Err("Verification cancelled.".into()), None, models);
        }
        let body = json!({
            "model": candidate,
            "max_tokens": 6,
            "messages": [{"role": "user", "content": "ping"}]
        });
        match http_post_json(&endpoint, provider, api_key, &body, cancel, false, 18) {
            Ok(resp) => {
                if let Some(err) = resp.get("error") {
                    let msg = err.get("message").and_then(Value::as_str).or_else(|| err.as_str()).unwrap_or("API error");
                    let lower = msg.to_ascii_lowercase();
                    // If the API key itself is invalid/unauthorized, stop immediately!
                    if lower.contains("invalid api key")
                        || lower.contains("incorrect api key")
                        || lower.contains("unauthorized")
                        || lower.contains("authentication")
                        || lower.contains("invalid x-api-key")
                        || lower.contains("insufficient_quota")
                        || lower.contains("insufficient balance")
                    {
                        return (Err(msg.to_string()), None, models);
                    }
                    last_err = msg.to_string();
                    continue;
                }
                if resp.get("choices").is_some() || resp.get("content").is_some() || resp.get("id").is_some() {
                    let actual_model = resp.get("model").and_then(Value::as_str).unwrap_or(candidate.as_str()).to_string();
                    return (Ok(format!("{} ({actual_model})", provider.label())), Some(candidate.clone()), models);
                }
            }
            Err(e) => {
                last_err = e;
            }
        }
    }

    (Err(last_err), None, models)
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_api_models_list(provider: ApiProvider, base_url: &str, api_key: &str, cancel: &AtomicBool) -> Result<Vec<String>, String> {
    let models_url = format!("{base_url}/models");
    let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut args: Vec<String> = vec!["-sS".into(), "--max-time".into(), "8".into(), models_url];
    if provider == ApiProvider::Anthropic {
        args.push("-H".into());
        args.push(format!("x-api-key: {api_key}"));
        args.push("-H".into());
        args.push("anthropic-version: 2023-06-01".into());
    } else if !api_key.is_empty() {
        args.push("-H".into());
        args.push(format!("Authorization: Bearer {api_key}"));
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let out = run_process_cancellable_with_stdin(curl_bin, &arg_refs, None, None, cancel, false, 10)?;
    let val: Value = serde_json::from_str(&out).map_err(|e| e.to_string())?;
    let ids = val
        .get("data")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(|item| item.get("id").and_then(Value::as_str)).map(str::to_string).collect::<Vec<_>>())
        .unwrap_or_default();
    Ok(filter_chat_models(ids))
}

/// Discover available models from `agy models`.
#[cfg(not(target_arch = "wasm32"))]
fn discover_agy_models(cmd_name: &str, cancel: &AtomicBool) -> Option<Vec<(String, String)>> {
    let out = run_process_cancellable_with_stdin(cmd_name, &["models"], None, None, cancel, true, 8).ok()?;
    let mut list = Vec::new();
    for line in out.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((id, name)) = trimmed.split_once('\t') {
            let id = id.trim().to_string();
            let name = name.trim().to_string();
            if !id.is_empty() {
                list.push((id, if name.is_empty() { "Model".into() } else { name }));
            }
        } else {
            let mut parts = trimmed.splitn(2, char::is_whitespace);
            if let Some(id) = parts.next() {
                let name = parts.next().unwrap_or(id).trim();
                list.push((id.to_string(), name.to_string()));
            }
        }
    }
    // Prioritize fast Flash & Argon models at the top of the list
    list.sort_by_key(|(id, _)| {
        if id.starts_with("gemini-3.8-flash-low") {
            0
        } else if id.starts_with("gemini-3.8-flash") {
            1
        } else if id.starts_with("gemini-4-argon-low") {
            2
        } else if id.starts_with("gemini-4-argon") {
            3
        } else if id.starts_with("gpt-") {
            4
        } else {
            5
        }
    });
    if list.is_empty() { None } else { Some(list) }
}

/// Discover available Codex CLI models from `~/.codex/models_cache.json` and `~/.codex/config.toml`.
#[cfg(not(target_arch = "wasm32"))]
fn discover_codex_models_from_disk() -> Option<Vec<(String, String)>> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let codex_dir = std::path::PathBuf::from(home).join(".codex");
    let mut list: Vec<(String, String)> = Vec::new();

    // 1. Check active model in `~/.codex/config.toml`
    let cfg_path = codex_dir.join("config.toml");
    if let Ok(cfg_text) = std::fs::read_to_string(&cfg_path) {
        for line in cfg_text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("model") {
                let rest = rest.trim_start();
                if let Some(val_part) = rest.strip_prefix('=') {
                    let slug = val_part.trim().trim_matches('"').trim_matches('\'').trim();
                    if !slug.is_empty() {
                        list.push((slug.to_string(), format!("{} (Configured)", slug.to_uppercase())));
                    }
                }
            }
        }
    }

    // 2. Read `~/.codex/models_cache.json` for official listed models (`gpt-5.5`, `gpt-5.4-mini`, etc.)
    let cache_path = codex_dir.join("models_cache.json");
    if let Ok(raw) = std::fs::read_to_string(&cache_path)
        && let Ok(parsed) = serde_json::from_str::<Value>(&raw)
        && let Some(models) = parsed.get("models").and_then(Value::as_array)
    {
        for m in models {
            let slug = m.get("slug").and_then(Value::as_str).unwrap_or("").trim();
            if slug.is_empty() {
                continue;
            }
            let vis = m.get("visibility").and_then(Value::as_str).unwrap_or("list");
            if vis == "hide" {
                continue;
            }
            let display = m.get("display_name").and_then(Value::as_str).unwrap_or(slug).trim();
            let desc = m.get("description").and_then(Value::as_str).unwrap_or("").trim();
            let label = if !desc.is_empty() { format!("{display} — {desc}") } else { display.to_string() };
            if let Some(existing) = list.iter_mut().find(|(id, _)| id == slug) {
                existing.1 = label;
            } else {
                list.push((slug.to_string(), label));
            }
        }
    }

    if list.is_empty() { None } else { Some(list) }
}

/// Verify whether a specific model on a Terminal CLI Agent can actually be used by the user's account!
#[cfg(not(target_arch = "wasm32"))]
fn verify_cli_model_usable(cli: CliAgent, cmd_name: &str, model_id: &str, cancel: &AtomicBool) -> Result<String, String> {
    let work_dir = isolated_agent_workspace();
    let probe_prompt = "Reply only with the single word OK";

    let (args, stdin_bytes): (Vec<String>, Option<&[u8]>) = match cli {
        CliAgent::Agy => {
            ensure_clean_agy_mcp_config();
            let mut a = vec![
                "-p".to_string(),
                probe_prompt.to_string(),
                "--disable-slash-commands".to_string(),
                "--dangerously-skip-permissions".to_string(),
                "--print-timeout".to_string(),
                "45s".to_string(),
            ];
            if !model_id.is_empty() && model_id != "default" {
                a.push("--model".to_string());
                a.push(model_id.to_string());
            }
            (a, None)
        }
        CliAgent::Claude => {
            let mut a = vec![
                "-p".to_string(),
                "--tools".to_string(),
                "".to_string(),
                "--strict-mcp-config".to_string(),
                "--no-session-persistence".to_string(),
            ];
            if !model_id.is_empty() && model_id != "default" {
                a.push("--model".to_string());
                a.push(model_id.to_string());
            }
            (a, Some(probe_prompt.as_bytes()))
        }
        CliAgent::Codex => {
            let mut a = vec![
                "exec".to_string(),
                "-c".to_string(),
                "mcp_servers={}".to_string(),
                "-c".to_string(),
                "model_reasoning_effort=\"low\"".to_string(),
                "--skip-git-repo-check".to_string(),
            ];
            if !model_id.is_empty() && model_id != "default" {
                a.push("-m".to_string());
                a.push(model_id.to_string());
            }
            a.push("-".to_string());
            (a, Some(probe_prompt.as_bytes()))
        }
        CliAgent::Custom => (vec!["--version".to_string()], None),
    };

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let out = run_process_cancellable_with_stdin(cmd_name, &arg_refs, stdin_bytes, work_dir.as_deref(), cancel, false, 60)?;

    if let Some(err_msg) = detect_cli_output_error(&out) {
        return Err(err_msg);
    }

    Ok(format!("{cmd_name} ({model_id}) [OK Verified]"))
}

/// Inspect CLI stdout/stderr for model access errors, quota errors, or print timeouts even if exit code was 0.
#[cfg(not(target_arch = "wasm32"))]
fn detect_cli_output_error(output: &str) -> Option<String> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return Some("CLI agent returned an empty response.".into());
    }
    for line in trimmed.lines() {
        let l = line.trim();
        let lower = l.to_ascii_lowercase();
        if lower.starts_with("error:")
            || lower.starts_with("agy_error:")
            || lower.contains("individual quota reached")
            || lower.contains("resource_exhausted")
            || lower.contains("there's an issue with the selected model")
            || lower.contains("may not have access to it")
            || lower.contains("model is not supported")
            || lower.contains("not logged into")
            || lower.contains("print timeout after")
            || lower.contains("reconnecting... 3/5")
        {
            return Some(l.to_string());
        }
    }
    None
}

/// Ensure `~/.gemini/config/mcp_config.json` does not contain a hanging `unity-mcp` entry when Unity is not running.
#[cfg(not(target_arch = "wasm32"))]
fn ensure_clean_agy_mcp_config() {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    let Some(home_dir) = home.map(std::path::PathBuf::from) else {
        return;
    };
    let cfg_path = home_dir.join(".gemini").join("config").join("mcp_config.json");
    let Ok(raw) = std::fs::read_to_string(&cfg_path) else {
        return;
    };
    let Ok(mut val) = serde_json::from_str::<Value>(&raw) else {
        return;
    };
    let Some(servers) = val.get_mut("mcpServers").and_then(Value::as_object_mut) else {
        return;
    };
    // Remove any disabled entries or `unity-mcp` entry that blocks `agy` print mode when Unity is closed.
    let keys_to_remove: Vec<String> = servers
        .iter()
        .filter_map(|(k, v)| {
            let is_disabled = v.get("disabled").and_then(Value::as_bool) == Some(true);
            let is_unity = k == "unity-mcp";
            if is_disabled || is_unity { Some(k.clone()) } else { None }
        })
        .collect();
    if !keys_to_remove.is_empty() {
        for k in keys_to_remove {
            servers.remove(&k);
        }
        if let Ok(updated) = serde_json::to_string_pretty(&val) {
            let _ = std::fs::write(&cfg_path, updated);
        }
    }
}

/// Return an isolated empty directory in `%TEMP%\gridcraft_ai_workspace` so CLI agents never scan
/// the 100k-line Rust repository or load repository-level `AGENTS.md` / `CLAUDE.md` files!
#[cfg(not(target_arch = "wasm32"))]
fn isolated_agent_workspace() -> Option<std::path::PathBuf> {
    let dir = std::env::temp_dir().join("gridcraft_ai_workspace");
    let _ = std::fs::create_dir_all(&dir);
    Some(dir)
}

fn send_chat_prompt(state: &mut AiAgentState, prompt: String, ctx: egui::Context) {
    state.sync_active_backend();
    let backend = state.active_chat_backend;

    state.error_notification = None;
    state.messages.push(ChatEntry::User(prompt.clone()));
    state.cancel_flag.store(true, Ordering::SeqCst);
    let cancel = Arc::new(AtomicBool::new(false));
    state.cancel_flag = Arc::clone(&cancel);
    state.running = true;

    let (tx, rx) = channel();
    state.worker_rx = Some(rx);

    let provider = state.provider;
    let base_url = if provider == ApiProvider::Custom {
        state.base_url.trim().trim_end_matches('/').to_string()
    } else {
        provider.default_base_url().to_string()
    };
    let api_key = state.api_key.trim().to_string();
    let model = state.model.trim().to_string();

    let cli_agent = state.cli_agent;
    let custom_cli = state.custom_cli_cmd.trim().to_string();
    let cli_model = state.cli_model.trim().to_string();

    let ollama_base = normalize_local_url(state.ollama_url.trim().trim_end_matches('/'));
    let ollama_model = state.ollama_model.trim().to_string();

    let control_tx = state.control_tx.clone();
    let control_port = state.control_port;

    let history: Vec<(String, String)> = state
        .messages
        .iter()
        .filter_map(|m| match m {
            ChatEntry::User(u) => Some(("user".into(), u.clone())),
            ChatEntry::Assistant(a) => Some(("assistant".into(), a.clone())),
            _ => None,
        })
        .collect();

    #[cfg(not(target_arch = "wasm32"))]
    {
        let full_tool_access = state.full_tool_access;
        let include_command_catalog = state.include_command_catalog;
        let allow_ui_tools = state.allow_ui_tools;
        let _ = std::thread::Builder::new().name("ai-chat-worker".into()).spawn(move || {
            match backend {
                AgentMode::Api => {
                    if !check_internet_online(Some(&base_url)) {
                        let _ = tx.send(WorkerEvent::ConnectionLost { mode: AgentMode::Api, reason: "No internet connection detected".into() });
                        ctx.request_repaint();
                        return;
                    }
                    run_api_agent_loop(
                        AgentMode::Api,
                        provider,
                        &base_url,
                        &api_key,
                        &model,
                        history,
                        control_tx,
                        &tx,
                        &cancel,
                        &ctx,
                        false,
                        full_tool_access,
                        include_command_catalog,
                        allow_ui_tools,
                    );
                }
                AgentMode::Ollama => {
                    if !check_endpoint_reachable(&ollama_base) {
                        let _ = tx.send(WorkerEvent::ConnectionLost {
                            mode: AgentMode::Ollama,
                            reason: format!("Ollama server at {ollama_base} is unreachable"),
                        });
                        ctx.request_repaint();
                        return;
                    }
                    let ollama_v1 = format!("{ollama_base}/v1");
                    run_api_agent_loop(
                        AgentMode::Ollama,
                        ApiProvider::Custom,
                        &ollama_v1,
                        "ollama",
                        &ollama_model,
                        history,
                        control_tx,
                        &tx,
                        &cancel,
                        &ctx,
                        true,
                        full_tool_access,
                        include_command_catalog,
                        allow_ui_tools,
                    );
                }
                AgentMode::Terminal => {
                    if !check_internet_online(None) {
                        let _ = tx.send(WorkerEvent::ConnectionLost { mode: AgentMode::Terminal, reason: "No internet connection detected".into() });
                        ctx.request_repaint();
                        return;
                    }
                    let cmd = if cli_agent == CliAgent::Custom { custom_cli } else { cli_agent.cmd_name().to_string() };
                    run_cli_agent_turn(
                        cli_agent,
                        &cmd,
                        &cli_model,
                        &prompt,
                        control_tx,
                        control_port,
                        &tx,
                        &cancel,
                        &ctx,
                        full_tool_access,
                        include_command_catalog,
                        allow_ui_tools,
                    );
                }
            }
            let _ = tx.send(WorkerEvent::Finished);
            ctx.request_repaint();
        });
    }
}

// ============================================================================
// In-Process MCP Tool Definitions, Live Grid/Table Context & Smart Chart Engine
// ============================================================================

const SYSTEM_PROMPT: &str = "You are the built-in AI Agent inside GridCraft, a Rust-native spreadsheet. \
Work on the open workbook with the provided tools. Follow the ROLE AND GROUND RULES above: work orders \
are executed, questions about the workbook are answered from tool results, GridCraft or spreadsheet \
questions get a 1-3 line answer plus an offer to do it, and out-of-topic questions are declined. \
When the sheet is empty and the user asks for a report, list or data, build it yourself with write_range, \
formulas, number formats, a table and (when asked) a chart. Keep replies short plain text: no asterisks, \
no markdown tables, at most 3 lines. Always English. Never fabricate command ids, values or results.\
If native function calling is unavailable, emit tool calls as: \
<gridcraft_tool>{\"name\":\"write_range\",\"arguments\":{\"range\":\"A1\",\"values\":[[\"Product\",\"Qty\",\"Price\",\"Total\"],[\"Item A\",10,150000,\"=B2*C2\"]]}}</gridcraft_tool>";

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct ColStyleBlueprint {
    header_text: String,
    header_style: gridcraft_engine::model::Style,
    even_data_style: gridcraft_engine::model::Style,
    odd_data_style: gridcraft_engine::model::Style,
    totals_style: Option<gridcraft_engine::model::Style>,
    data_formula_template: Option<(u32, String)>,
    totals_label: Option<String>,
    totals_formula_fn: Option<String>,
    is_numeric_col: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct SecondaryTableSnapshot {
    range: RangeRef,
    headers: Vec<String>,
    data_rows: Vec<Vec<gridcraft_engine::core::Value>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct TableStyleBlueprint {
    formal_table: Option<gridcraft_engine::model::Table>,
    orig_range: RangeRef,
    header_row: Option<u32>,
    data_start_row: u32,
    data_end_row: u32,
    totals_row: Option<u32>,
    col_blueprints: Vec<ColStyleBlueprint>,
    sparkline_templates: Vec<(u32, u32, u32, gridcraft_engine::model::Sparkline)>,
    cond_format_templates: Vec<(u32, u32, gridcraft_engine::model::CondFormat)>,
    secondary_tables: Vec<SecondaryTableSnapshot>,
}

/// Handle custom AI-specific control methods directly on `SheetApp` (on the UI thread):
/// - `"ai.capture_blueprint"`: snapshots Table 1's exact styles, number formats, formulas, sparklines,
///   conditional formats, and secondary tables before a turn modifies the sheet.
/// - `"ai.sync_table_format"`: ensures that when tables are merged or expanded, 100% of rows and columns
///   follow Table 1's exact formatting, number formats, formulas, sparklines, and conditional formatting,
///   and clears any leftover cells/styles from absorbed secondary tables.
/// - `"ai.deep_clear"`: completely deletes cell data, cell background colors/styles, formal tables,
///   chart widgets, sparklines, conditional formats, shapes, images, and merges.
/// - `"ai.cleanup_orphaned_charts"`: removes any chart widgets or formal tables whose underlying cells
///   have been cleared, and runs a full sheet deep-clear if no data cells remain on the sheet.
#[cfg(not(target_arch = "wasm32"))]
fn handle_ai_custom_control(app: &mut SheetApp, method: &str, params: &Value) -> Option<Value> {
    match method {
        "ai.capture_blueprint" => {
            app.ai_agent.table_blueprint = capture_table_blueprint_from_app(app);
            Some(json!({"ok": true, "result": {"captured": app.ai_agent.table_blueprint.is_some()}}))
        }
        "ai.sync_table_format" => {
            let merge_requested = params.get("merge_requested").and_then(Value::as_bool).unwrap_or(false);
            let synced = apply_table_blueprint_on_app(app, merge_requested);
            Some(json!({"ok": true, "result": {"synced": synced}}))
        }
        "ai.deep_clear" => {
            let all = params.get("all").and_then(Value::as_bool).unwrap_or(false);
            let range_str = params.get("range").and_then(Value::as_str).unwrap_or("").trim();
            if all || range_str.is_empty() || range_str.eq_ignore_ascii_case("all") {
                deep_clear_on_app(app, None);
                return Some(json!({"ok": true, "result": {"cleared": true, "all": true}}));
            }
            let Some(range) = RangeRef::parse(split_sheet(range_str).1) else {
                // A range the model meant to clear but that cannot be parsed must never fall through
                // to "clear the whole sheet": report it so the agent retries with a valid A1 range.
                return Some(json!({
                    "ok": false,
                    "error": format!(
                        "invalid range `{range_str}`: pass an A1 range like A1:C20 (or all=true to clear the sheet)"
                    )
                }));
            };
            deep_clear_on_app(app, Some(range));
            Some(json!({"ok": true, "result": {"cleared": true, "range": range_str}}))
        }
        "ai.cleanup_orphaned_charts" => {
            let cleaned = cleanup_orphaned_charts_on_app(app);
            Some(json!({"ok": true, "result": {"cleaned": cleaned}}))
        }
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn find_sheet_table_blocks(sh: &gridcraft_engine::model::Sheet) -> Vec<(RangeRef, bool, Option<usize>)> {
    let Some(used) = sh.used_range() else {
        return Vec::new();
    };
    let mut blocks: Vec<(RangeRef, bool, Option<usize>)> = Vec::new();
    for (idx, ft) in sh.tables.iter().enumerate() {
        blocks.push((ft.range, ft.totals_row, Some(idx)));
    }
    let formal_ranges: Vec<RangeRef> = blocks.iter().map(|(r, _, _)| *r).collect();
    let is_row_in_formal = |row: u32| -> bool { formal_ranges.iter().any(|r| row >= r.start.row && row <= r.end.row) };

    let cell_has_content =
        |r: u32, c: u32| -> bool { sh.cell(CellRef::new(r, c)).is_some_and(|cell| !cell.value.is_empty() || cell.formula.is_some()) };

    let mut cur: Option<(u32, u32, u32, u32)> = None;
    for row in used.start.row..=used.end.row {
        if is_row_in_formal(row) {
            if let Some((r0, r1, c0, c1)) = cur.take()
                && r1 > r0
            {
                blocks.push((RangeRef::new(CellRef::new(r0, c0), CellRef::new(r1, c1)), false, None));
            }
            continue;
        }
        let mut non_empty_cols = Vec::new();
        for col in used.start.col..=used.end.col {
            if cell_has_content(row, col) {
                non_empty_cols.push(col);
            }
        }
        if non_empty_cols.len() >= 2 {
            let rc0 = non_empty_cols.first().copied().unwrap_or(used.start.col);
            let rc1 = non_empty_cols.last().copied().unwrap_or(used.end.col);
            match &mut cur {
                Some((_, r1, c0, c1)) => {
                    *r1 = row;
                    *c0 = (*c0).min(rc0);
                    *c1 = (*c1).max(rc1);
                }
                None => {
                    cur = Some((row, row, rc0, rc1));
                }
            }
        } else if let Some((r0, r1, c0, c1)) = cur.take()
            && r1 > r0
        {
            blocks.push((RangeRef::new(CellRef::new(r0, c0), CellRef::new(r1, c1)), false, None));
        }
    }
    if let Some((r0, r1, c0, c1)) = cur.take()
        && r1 > r0
    {
        blocks.push((RangeRef::new(CellRef::new(r0, c0), CellRef::new(r1, c1)), false, None));
    }

    // Sort so formal tables and wider/larger tables come first (Table 1 first, secondary tables after)
    blocks.sort_by_key(|(r, _, ft_idx)| {
        (std::cmp::Reverse(ft_idx.is_some()), std::cmp::Reverse(r.width() >= 3), std::cmp::Reverse(r.count()), r.start.row)
    });
    blocks
}

#[cfg(not(target_arch = "wasm32"))]
fn is_summary_row_on_sheet(sh: &gridcraft_engine::model::Sheet, row: u32, c0: u32, c1: u32) -> bool {
    for col in c0..=c1 {
        if let Some(cell) = sh.cell(CellRef::new(row, col))
            && let gridcraft_engine::core::Value::Text(s) = &cell.value
        {
            let low = s.trim().to_ascii_lowercase();
            if matches!(low.as_str(), "total" | "jumlah" | "subtotal" | "grand total" | "rata-rata" | "rata rata" | "average")
                || low.starts_with("total ")
                || low.starts_with("jumlah ")
            {
                return true;
            }
        }
    }
    false
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_table_blueprint_from_app(app: &SheetApp) -> Option<TableStyleBlueprint> {
    let doc = app.session.doc().ok()?;
    let sh = doc.wb.sheet(doc.wb.active_sheet())?;
    let blocks = find_sheet_table_blocks(sh);
    let &(orig_range, formal_totals, ft_idx) = blocks.first()?;

    let formal_table = ft_idx.and_then(|i| sh.tables.get(i).cloned());

    let mut text_headers = 0usize;
    let mut non_empty_top = 0usize;
    for col in orig_range.start.col..=orig_range.end.col {
        if let Some(cell) = sh.cell(CellRef::new(orig_range.start.row, col)) {
            match &cell.value {
                gridcraft_engine::core::Value::Text(s) if !s.trim().is_empty() => {
                    text_headers += 1;
                    non_empty_top += 1;
                }
                gridcraft_engine::core::Value::Empty if cell.formula.is_none() => {}
                _ => non_empty_top += 1,
            }
        }
    }
    let has_header = orig_range.height() >= 2 && text_headers > 0 && text_headers * 2 >= non_empty_top;
    let header_row = has_header.then_some(orig_range.start.row);
    let data_start_row = if has_header { orig_range.start.row + 1 } else { orig_range.start.row };
    let has_totals = formal_totals
        || (orig_range.end.row > data_start_row && is_summary_row_on_sheet(sh, orig_range.end.row, orig_range.start.col, orig_range.end.col));
    let totals_row = (has_totals && orig_range.end.row > data_start_row).then_some(orig_range.end.row);
    let data_end_row = if totals_row.is_some() { orig_range.end.row - 1 } else { orig_range.end.row };

    let mut col_blueprints = Vec::new();
    for col in orig_range.start.col..=orig_range.end.col {
        let header_text = header_row.and_then(|hr| sh.cell(CellRef::new(hr, col))).map(|c| c.value.display()).unwrap_or_default();
        let header_style = doc.wb.styles.get(sh.style_id(CellRef::new(header_row.unwrap_or(data_start_row), col))).clone();
        let even_data_style = doc.wb.styles.get(sh.style_id(CellRef::new(data_start_row, col))).clone();
        let second_data_row = (data_start_row + 1).min(data_end_row);
        let odd_data_style = doc.wb.styles.get(sh.style_id(CellRef::new(second_data_row, col))).clone();
        let totals_style = totals_row.map(|tr| doc.wb.styles.get(sh.style_id(CellRef::new(tr, col))).clone());

        let data_formula_template =
            sh.cell(CellRef::new(data_start_row, col)).and_then(|c| c.formula.as_ref()).map(|f| (data_start_row, f.text.clone()));

        let mut totals_label = None;
        let mut totals_formula_fn = None;
        if let Some(tr) = totals_row
            && let Some(tcell) = sh.cell(CellRef::new(tr, col))
        {
            if let Some(f) = &tcell.formula {
                if let Some((fn_name, _)) = f.text.split_once('(') {
                    let clean_fn = fn_name.trim().to_ascii_uppercase();
                    if matches!(clean_fn.as_str(), "SUM" | "AVERAGE" | "COUNT" | "MIN" | "MAX" | "SUBTOTAL") {
                        totals_formula_fn = Some(clean_fn);
                    }
                }
            } else if let gridcraft_engine::core::Value::Text(s) = &tcell.value
                && !s.trim().is_empty()
            {
                totals_label = Some(s.to_string());
            }
        }

        let is_numeric_col = data_formula_template.is_some()
            || !matches!(even_data_style.num_fmt.as_str(), "General" | "" | "@")
            || (data_start_row..=data_end_row)
                .any(|r| sh.cell(CellRef::new(r, col)).is_some_and(|c| matches!(c.value, gridcraft_engine::core::Value::Number(_))));

        col_blueprints.push(ColStyleBlueprint {
            header_text,
            header_style,
            even_data_style,
            odd_data_style,
            totals_style,
            data_formula_template,
            totals_label,
            totals_formula_fn,
            is_numeric_col,
        });
    }

    let mut sparkline_templates = Vec::new();
    for sp in &sh.sparklines {
        if sp.cell.row == data_start_row
            && sp.cell.col >= orig_range.start.col
            && sp.cell.col <= orig_range.end.col
            && let Some(src_r) = RangeRef::parse(split_sheet(&sp.source).1)
            && src_r.start.col >= orig_range.start.col
            && src_r.end.col <= orig_range.end.col
        {
            sparkline_templates.push((
                sp.cell.col - orig_range.start.col,
                src_r.start.col - orig_range.start.col,
                src_r.end.col - orig_range.start.col,
                sp.clone(),
            ));
        }
    }

    let mut cond_format_templates = Vec::new();
    for cf in &sh.cond_formats {
        if let Some(r0) = cf.ranges.first()
            && r0.start.col >= orig_range.start.col
            && r0.end.col <= orig_range.end.col
        {
            cond_format_templates.push((r0.start.col - orig_range.start.col, r0.end.col - orig_range.start.col, cf.clone()));
        }
    }

    let mut secondary_tables = Vec::new();
    for &(sec_r, sec_totals, _) in blocks.iter().skip(1) {
        let mut sec_headers = Vec::new();
        let mut first_row_matches_header = false;
        for col in sec_r.start.col..=sec_r.end.col {
            let txt = sh.cell(CellRef::new(sec_r.start.row, col)).map(|c| c.value.display()).unwrap_or_default();
            if col_blueprints.iter().any(|cb| !cb.header_text.is_empty() && cb.header_text.eq_ignore_ascii_case(txt.trim())) {
                first_row_matches_header = true;
            }
            sec_headers.push(txt);
        }
        let sec_start_row = if first_row_matches_header && sec_r.height() >= 2 { sec_r.start.row + 1 } else { sec_r.start.row };
        let sec_has_tot = sec_totals || (sec_r.end.row > sec_start_row && is_summary_row_on_sheet(sh, sec_r.end.row, sec_r.start.col, sec_r.end.col));
        let sec_end_row = if sec_has_tot && sec_r.end.row > sec_start_row { sec_r.end.row - 1 } else { sec_r.end.row };
        let mut data_rows = Vec::new();
        for r in sec_start_row..=sec_end_row {
            let mut row_vals = Vec::new();
            for c in sec_r.start.col..=sec_r.end.col {
                let v = sh.cell(CellRef::new(r, c)).map(|cell| cell.value.clone()).unwrap_or(gridcraft_engine::core::Value::Empty);
                row_vals.push(v);
            }
            data_rows.push(row_vals);
        }
        secondary_tables.push(SecondaryTableSnapshot { range: sec_r, headers: sec_headers, data_rows });
    }

    Some(TableStyleBlueprint {
        formal_table,
        orig_range,
        header_row,
        data_start_row,
        data_end_row,
        totals_row,
        col_blueprints,
        sparkline_templates,
        cond_format_templates,
        secondary_tables,
    })
}

/// Shift row numbers inside a row-level formula template (e.g. `SUM(C5:F5)` or `(F5-C5)/C5`)
/// from `from_row` (0-indexed) to `to_row` (0-indexed).
#[cfg(not(target_arch = "wasm32"))]
fn shift_row_formula(formula_text: &str, from_row: u32, to_row: u32) -> String {
    let from_1 = (from_row + 1).to_string();
    let to_1 = (to_row + 1).to_string();
    let bytes = formula_text.as_bytes();
    let mut out = String::with_capacity(formula_text.len() + 4);
    let mut i = 0;
    while i < bytes.len() {
        let prev_ok = if i == 0 {
            true
        } else {
            let p = bytes[i - 1] as char;
            !p.is_ascii_alphanumeric() && p != '_'
        };
        if prev_ok {
            let mut j = i;
            if j < bytes.len() && bytes[j] == b'$' {
                j += 1;
            }
            let col_start = j;
            while j < bytes.len() && (bytes[j] as char).is_ascii_alphabetic() {
                j += 1;
            }
            let col_len = j - col_start;
            if (1..=3).contains(&col_len) {
                let mut k = j;
                if k < bytes.len() && bytes[k] == b'$' {
                    k += 1;
                }
                if formula_text[k..].starts_with(&from_1) {
                    let after_idx = k + from_1.len();
                    let next_ok = if after_idx >= bytes.len() {
                        true
                    } else {
                        let nx = bytes[after_idx] as char;
                        !nx.is_ascii_digit() && nx != '('
                    };
                    if next_ok {
                        out.push_str(&formula_text[i..k]);
                        out.push_str(&to_1);
                        i = after_idx;
                        continue;
                    }
                }
            }
        }
        if let Some(ch) = formula_text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// Apply Table 1's captured style blueprint to the merged/expanded table on the active sheet so
/// 100% of rows and columns match Table 1's font, fill, border, alignment, numberFormat, formulas,
/// conditional formatting, and sparklines!
#[cfg(not(target_arch = "wasm32"))]
fn apply_table_blueprint_on_app(app: &mut SheetApp, merge_requested: bool) -> bool {
    let Some(bp) = app.ai_agent.table_blueprint.clone() else {
        return false;
    };
    let Ok(doc) = app.session.doc_mut() else {
        return false;
    };
    let wb = Arc::make_mut(&mut doc.wb);
    let active_idx = wb.active_sheet();
    let Some(sh) = wb.sheet(active_idx) else {
        return false;
    };

    let cell_has_val = |sheet: &gridcraft_engine::model::Sheet, r: u32, c: u32| -> bool {
        sheet.cell(CellRef::new(r, c)).is_some_and(|cell| !cell.value.is_empty() || cell.formula.is_some())
    };

    // Determine current bounds of the primary table starting near `bp.orig_range.start`
    let Some(used) = sh.used_range() else {
        return false;
    };
    let start_row = bp.orig_range.start.row.min(used.end.row);
    let start_col = bp.orig_range.start.col.min(used.end.col);
    let mut end_col = bp.orig_range.end.col;
    for c in bp.orig_range.start.col..=used.end.col {
        if cell_has_val(sh, start_row, c) || cell_has_val(sh, bp.data_start_row, c) {
            end_col = end_col.max(c);
        }
    }

    // Scan contiguous non-empty rows starting at `start_row`
    let mut end_row = start_row;
    let mut r_cursor = start_row;
    while r_cursor <= used.end.row {
        let row_non_empty = (start_col..=end_col).any(|c| cell_has_val(sh, r_cursor, c));
        if row_non_empty {
            end_row = r_cursor;
            r_cursor += 1;
        } else {
            // Check if the AI left a 1-row gap before appending rows right under `bp.orig_range`
            let next_row_in_same_cols = r_cursor < used.end.row && (start_col..=end_col).filter(|&c| cell_has_val(sh, r_cursor + 1, c)).count() >= 3;
            if next_row_in_same_cols && r_cursor <= bp.orig_range.end.row + 1 {
                r_cursor += 1;
            } else {
                break;
            }
        }
    }

    // If `merge_requested` is true and the AI did not yet append the secondary table's rows into Table 1,
    // automatically append compatible rows from `bp.secondary_tables`!
    let orig_data_count = bp.data_end_row.saturating_sub(bp.data_start_row) + 1;
    let cur_total_rows = end_row.saturating_sub(start_row) + 1;
    let table_did_not_grow = cur_total_rows <= bp.orig_range.height();

    if merge_requested
        && !bp.secondary_tables.is_empty()
        && table_did_not_grow
        && let Some(sh_mut) = wb.sheet_mut(active_idx)
    {
        let mut insert_row = bp.data_end_row + 1;
        for sec in &bp.secondary_tables {
            for sec_row_vals in &sec.data_rows {
                let non_empty_count = sec_row_vals.iter().filter(|v| !v.is_empty()).count();
                if non_empty_count == 0 {
                    continue;
                }
                for (sec_ci, val) in sec_row_vals.iter().enumerate() {
                    if val.is_empty() {
                        continue;
                    }
                    // Map secondary column to primary table column by header name or position
                    let sec_hdr = sec.headers.get(sec_ci).map(|s| s.trim()).unwrap_or("");
                    let target_col_off = bp
                        .col_blueprints
                        .iter()
                        .position(|cb| !sec_hdr.is_empty() && !cb.header_text.is_empty() && cb.header_text.eq_ignore_ascii_case(sec_hdr))
                        .unwrap_or(sec_ci);
                    let target_col = start_col + target_col_off as u32;
                    if target_col <= end_col {
                        sh_mut.set_cell(CellRef::new(insert_row, target_col), gridcraft_engine::model::Cell::value(val.clone()));
                    }
                }
                end_row = end_row.max(insert_row);
                insert_row += 1;
            }
        }
    }

    // Remove any middle `Total` / `Jumlah` summary row that got trapped between `data_start_row` and `end_row`
    // when new rows were appended below the original `Totals` row!
    if let Some(sh_mut) = wb.sheet_mut(active_idx) {
        let mut r = bp.data_start_row;
        while r < end_row {
            let is_mid_total = is_summary_row_on_sheet(sh_mut, r, start_col, end_col)
                || (bp.totals_row == Some(r)
                    && (start_col..=end_col).any(|c| {
                        sh_mut
                            .cell(CellRef::new(r, c))
                            .and_then(|cell| cell.formula.as_ref())
                            .is_some_and(|f| f.text.to_ascii_uppercase().starts_with("SUM("))
                    }));
            if is_mid_total {
                for shift_r in r..end_row {
                    for c in start_col..=end_col {
                        let next_cell = sh_mut.cell(CellRef::new(shift_r + 1, c)).cloned().unwrap_or_default();
                        sh_mut.set_cell(CellRef::new(shift_r, c), next_cell);
                    }
                }
                for c in start_col..=end_col {
                    sh_mut.set_cell(CellRef::new(end_row, c), gridcraft_engine::model::Cell::default());
                }
                end_row = end_row.saturating_sub(1);
            } else {
                r += 1;
            }
        }
    }

    // Determine `data_start_row`, `data_end_row`, and `totals_row` on the merged table
    let data_start_row = bp.data_start_row;
    let bottom_is_summary = wb.sheet(active_idx).is_some_and(|s| end_row > data_start_row && is_summary_row_on_sheet(s, end_row, start_col, end_col));

    let (data_end_row, totals_row) = if bottom_is_summary {
        (end_row - 1, Some(end_row))
    } else if bp.totals_row.is_some() {
        let new_tot_row = end_row + 1;
        end_row = new_tot_row;
        (new_tot_row - 1, Some(new_tot_row))
    } else {
        (end_row, None)
    };

    let target_range = RangeRef::new(CellRef::new(start_row, start_col), CellRef::new(end_row, end_col));

    // If tables were merged (or if secondary tables overlapped/were absorbed), clear any leftover cells & styles
    // from `bp.secondary_tables` that lie outside `target_range`!
    let merged_happened = merge_requested || (data_end_row.saturating_sub(data_start_row) + 1) > orig_data_count;
    if merged_happened && let Some(sh_mut) = wb.sheet_mut(active_idx) {
        for sec in &bp.secondary_tables {
            let sec_overlaps = sec.range.end.row >= target_range.start.row && sec.range.start.row <= target_range.end.row + 4;
            if merge_requested || sec_overlaps {
                for r in sec.range.start.row..=sec.range.end.row {
                    for c in sec.range.start.col..=sec.range.end.col {
                        let at = CellRef::new(r, c);
                        if !target_range.contains(at) {
                            sh_mut.set_cell(at, gridcraft_engine::model::Cell::default());
                        }
                    }
                }
            }
        }
    }

    // Intern all blueprint styles into `wb.styles` first so we have their `StyleId`s ready
    let fallback_cb = bp.col_blueprints.last().cloned();
    let mut resolved_col_styles = Vec::new();
    for col in start_col..=end_col {
        let col_off = (col - start_col) as usize;
        let hdr_txt = bp.header_row.and_then(|hr| wb.sheet(active_idx)?.cell(CellRef::new(hr, col))).map(|c| c.value.display()).unwrap_or_default();
        let cb = bp
            .col_blueprints
            .get(col_off)
            .or_else(|| bp.col_blueprints.iter().find(|x| !hdr_txt.is_empty() && x.header_text.eq_ignore_ascii_case(hdr_txt.trim())))
            .or(fallback_cb.as_ref());
        let Some(cb) = cb else {
            continue;
        };
        let hdr_id = wb.styles.intern(cb.header_style.clone());
        let even_id = wb.styles.intern(cb.even_data_style.clone());
        let odd_id = wb.styles.intern(cb.odd_data_style.clone());
        let tot_id = wb.styles.intern(cb.totals_style.clone().unwrap_or_else(|| cb.even_data_style.clone()));
        resolved_col_styles.push((col, cb.clone(), hdr_id, even_id, odd_id, tot_id));
    }

    // Apply styles, number formats, row formulas, totals formulas, formal table, cond_formats, and sparklines!
    if let Some(sh_mut) = wb.sheet_mut(active_idx) {
        let mut header_names = Vec::new();
        for (col, cb, hdr_id, even_id, odd_id, tot_id) in &resolved_col_styles {
            let col = *col;
            // 1. Header Row
            if let Some(hr) = bp.header_row {
                let at = CellRef::new(hr, col);
                let mut cell = sh_mut.cell(at).cloned().unwrap_or_default();
                if cell.value.is_empty() && !cb.header_text.is_empty() {
                    cell.value = gridcraft_engine::core::Value::text(cb.header_text.clone());
                }
                header_names.push(cell.value.display());
                cell.style = *hdr_id;
                sh_mut.set_cell(at, cell);
            } else {
                header_names.push(cb.header_text.clone());
            }

            // 2. Data Rows (`data_start_row..=data_end_row`)
            for row in data_start_row..=data_end_row {
                let at = CellRef::new(row, col);
                let st_id = if (row - data_start_row) % 2 == 0 { *even_id } else { *odd_id };
                let mut cell = sh_mut.cell(at).cloned().unwrap_or_default();

                // If this column has a row formula template (e.g. `=SUM(C5:F5)` or `=(F5-C5)/C5`)
                // and the cell is empty or was previously a Totals formula, populate the shifted row formula!
                let has_old_sum_range = cell.formula.as_ref().is_some_and(|f| {
                    let up = f.text.to_ascii_uppercase();
                    up.starts_with("SUM(") && !cb.data_formula_template.as_ref().is_some_and(|(_, t)| t.to_ascii_uppercase().starts_with("SUM("))
                });
                if (cell.value.is_empty() && cell.formula.is_none()) || has_old_sum_range {
                    if let Some((tmpl_row, tmpl_formula)) = &cb.data_formula_template {
                        let shifted = shift_row_formula(tmpl_formula, *tmpl_row, row);
                        cell.value = gridcraft_engine::core::Value::Empty;
                        cell.formula = Some(Arc::new(gridcraft_engine::model::Formula::new(&shifted)));
                    }
                } else if let Some((tmpl_row, tmpl_formula)) = &cb.data_formula_template
                    && cell.formula.is_some()
                {
                    // Ensure existing formula on shifted rows points to its own `row`
                    let shifted = shift_row_formula(tmpl_formula, *tmpl_row, row);
                    cell.formula = Some(Arc::new(gridcraft_engine::model::Formula::new(&shifted)));
                } else if cb.is_numeric_col
                    && cell.formula.is_none()
                    && let gridcraft_engine::core::Value::Text(s) = &cell.value
                {
                    let cleaned = s.trim().trim_start_matches(['$', '€', '£', '¥']).trim_start_matches("Rp").trim();
                    let is_pct = cleaned.ends_with('%');
                    let num_part = cleaned.trim_end_matches('%').replace(',', "");
                    if let Ok(mut parsed_n) = num_part.trim().parse::<f64>() {
                        if is_pct && cb.even_data_style.num_fmt.as_str().contains('%') && parsed_n.abs() > 1.0 {
                            parsed_n /= 100.0;
                        }
                        cell.value = gridcraft_engine::core::Value::Number(parsed_n);
                    }
                }

                cell.style = st_id;
                sh_mut.set_cell(at, cell);
            }

            // 3. Totals Row (`totals_row`)
            if let Some(tr) = totals_row {
                let at = CellRef::new(tr, col);
                let mut cell = sh_mut.cell(at).cloned().unwrap_or_default();
                if let Some(fn_name) = &cb.totals_formula_fn {
                    let col_let = col_to_letters(col);
                    let f_str = format!("{fn_name}({col_let}{}:{col_let}{})", data_start_row + 1, data_end_row + 1);
                    cell.value = gridcraft_engine::core::Value::Empty;
                    cell.formula = Some(Arc::new(gridcraft_engine::model::Formula::new(&f_str)));
                } else if let Some(lbl) = &cb.totals_label {
                    cell.value = gridcraft_engine::core::Value::text(lbl.clone());
                    cell.formula = None;
                }
                cell.style = *tot_id;
                sh_mut.set_cell(at, cell);
            }
        }

        // 4. Resize formal `Table` in `sh_mut.tables` to cover `target_range` and remove absorbed secondary tables
        let build_table_cols = |existing: &[gridcraft_engine::model::TableColumn], names: &[String]| -> Vec<gridcraft_engine::model::TableColumn> {
            names
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    if let Some(mut ex) = existing.get(i).cloned() {
                        if !n.is_empty() {
                            ex.name = n.clone();
                        }
                        ex
                    } else {
                        gridcraft_engine::model::TableColumn {
                            name: if n.is_empty() { format!("Column{}", i + 1) } else { n.clone() },
                            totals: gridcraft_engine::model::TotalsFn::None,
                            totals_label: None,
                            formula: None,
                            unique_name: None,
                            xml: None,
                        }
                    }
                })
                .collect()
        };

        if let Some(mut ft) = bp.formal_table.clone() {
            ft.range = target_range;
            ft.header_row = bp.header_row.is_some();
            ft.totals_row = totals_row.is_some();
            ft.columns = build_table_cols(&ft.columns, &header_names);
            sh_mut.tables = vec![ft];
        } else if !sh_mut.tables.is_empty() {
            let new_cols = build_table_cols(&sh_mut.tables[0].columns, &header_names);
            sh_mut.tables[0].range = target_range;
            sh_mut.tables[0].totals_row = totals_row.is_some();
            sh_mut.tables[0].columns = new_cols;
            if merged_happened {
                sh_mut.tables.truncate(1);
            }
        }

        // 5. Extend Conditional Formatting (`sh_mut.cond_formats`) to cover `data_start_row..=data_end_row`
        if !bp.cond_format_templates.is_empty() {
            let mut new_cfs = Vec::new();
            for (c0_off, c1_off, cf_tmpl) in &bp.cond_format_templates {
                let c0 = (start_col + *c0_off).min(end_col);
                let c1 = (start_col + *c1_off).min(end_col);
                let mut updated = cf_tmpl.clone();
                updated.ranges = vec![RangeRef::new(CellRef::new(data_start_row, c0), CellRef::new(data_end_row, c1))];
                new_cfs.push(updated);
            }
            sh_mut.cond_formats = new_cfs;
        }

        // 6. Extend Sparklines (`sh_mut.sparklines`) to cover every row in `data_start_row..=data_end_row`
        if !bp.sparkline_templates.is_empty() {
            let mut new_sps = Vec::new();
            for (sp_col_off, src_c0_off, src_c1_off, sp_tmpl) in &bp.sparkline_templates {
                let sp_col = start_col + *sp_col_off;
                let src_c0 = start_col + *src_c0_off;
                let src_c1 = start_col + *src_c1_off;
                if sp_col <= end_col && src_c1 <= end_col {
                    for row in data_start_row..=data_end_row {
                        let mut sp = sp_tmpl.clone();
                        sp.cell = CellRef::new(row, sp_col);
                        sp.source = RangeRef::new(CellRef::new(row, src_c0), CellRef::new(row, src_c1)).a1();
                        new_sps.push(sp);
                    }
                }
            }
            sh_mut.sparklines = new_sps;
        }
    }

    doc.calc.recalc_all(wb);
    true
}

/// Completely delete data, cell background colors/styles, formal tables, auto-filters, chart widgets,
/// sparklines, conditional formats, shapes, images, slicers, and merges on the active sheet (or within `range_opt`),
/// and reset multi-cell selection so no filter dropdown buttons or selection boxes remain!
#[cfg(not(target_arch = "wasm32"))]
fn deep_clear_on_app(app: &mut SheetApp, range_opt: Option<RangeRef>) {
    app.selected_chart = None;
    app.grid.filter_menu = None;
    app.editor = None;
    app.text_box_editor = None;
    let Ok(doc) = app.session.doc_mut() else {
        return;
    };
    let wb = Arc::make_mut(&mut doc.wb);
    let active_idx = wb.active_sheet();
    if let Some(r) = range_opt {
        if let Some(sh) = wb.sheet_mut(active_idx) {
            for row in r.start.row..=r.end.row {
                for col in r.start.col..=r.end.col {
                    sh.set_cell(CellRef::new(row, col), gridcraft_engine::model::Cell::default());
                }
            }
            if sh.autofilter.as_ref().is_some_and(|af| ranges_intersect(af.range, r)) {
                sh.autofilter = None;
            }
            sh.tables.retain(|t| !ranges_intersect(t.range, r));
            sh.cond_formats.retain(|cf| !cf.ranges.iter().any(|cr| ranges_intersect(*cr, r)));
            sh.sparklines.retain(|sp| {
                let src_hits = RangeRef::parse(split_sheet(&sp.source).1).is_some_and(|sr| ranges_intersect(sr, r));
                !r.contains(sp.cell) && !src_hits
            });
            sh.merges.retain(|m| !ranges_intersect(*m, r));
            sh.charts.retain(|ch| {
                let anchor_in_r = r.contains(ch.anchor.cell);
                let series_in_r = ch.series.iter().any(|s| RangeRef::parse(split_sheet(&s.values).1).is_some_and(|sr| ranges_intersect(sr, r)));
                !anchor_in_r && !series_in_r
            });
        }
        doc.calc.recalc_all(wb);
        cleanup_orphaned_charts_on_app(app);
    } else {
        if let Some(sh) = wb.sheet_mut(active_idx) {
            let name = sh.name.clone();
            let zoom = sh.zoom;
            let show_gridlines = sh.show_gridlines;
            let show_headings = sh.show_headings;
            let mut blank = gridcraft_engine::model::Sheet::new(name);
            blank.zoom = zoom;
            blank.show_gridlines = show_gridlines;
            blank.show_headings = show_headings;
            *sh = blank;
        }
        doc.selection = gridcraft_engine::Selection::at(CellRef::default());
        if active_idx < doc.sheet_selections.len() {
            doc.sheet_selections[active_idx] = gridcraft_engine::Selection::at(CellRef::default());
        }
        doc.calc.recalc_all(wb);
        app.ai_agent.table_blueprint = None;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ranges_intersect(a: RangeRef, b: RangeRef) -> bool {
    a.start.row <= b.end.row && a.end.row >= b.start.row && a.start.col <= b.end.col && a.end.col >= b.start.col
}

/// Remove any chart widgets, formal tables, auto-filters, sparklines, or conditional formats whose source cells
/// are now completely empty — and if the sheet has no data cells left at all, wipe all leftover
/// cell background colors/styles, auto-filters, selection boxes, and chart widgets!
#[cfg(not(target_arch = "wasm32"))]
fn cleanup_orphaned_charts_on_app(app: &mut SheetApp) -> bool {
    let sheet_has_any_data = {
        let Ok(doc) = app.session.doc() else {
            return false;
        };
        let Some(sh) = doc.wb.sheet(doc.wb.active_sheet()) else {
            return false;
        };
        if let Some(ur) = sh.used_range() {
            let mut found = false;
            'outer: for r in ur.start.row..=ur.end.row {
                for c in ur.start.col..=ur.end.col {
                    if let Some(cell) = sh.cell(CellRef::new(r, c))
                        && (!cell.value.is_empty() || cell.formula.is_some())
                    {
                        found = true;
                        break 'outer;
                    }
                }
            }
            found
        } else {
            false
        }
    };

    if !sheet_has_any_data {
        deep_clear_on_app(app, None);
        return true;
    }

    let Ok(doc) = app.session.doc_mut() else {
        return false;
    };
    let wb = Arc::make_mut(&mut doc.wb);
    let active_idx = wb.active_sheet();
    let Some(sh) = wb.sheet_mut(active_idx) else {
        return false;
    };

    let range_has_data = |sheet: &gridcraft_engine::model::Sheet, r: RangeRef| -> bool {
        for row in r.start.row..=r.end.row {
            for col in r.start.col..=r.end.col {
                if let Some(cell) = sheet.cell(CellRef::new(row, col))
                    && (!cell.value.is_empty() || cell.formula.is_some())
                {
                    return true;
                }
            }
        }
        false
    };

    let mut changed = false;
    if sh.autofilter.as_ref().is_some_and(|af| !range_has_data(sh, af.range)) {
        sh.autofilter = None;
        changed = true;
    }

    let mut keep_tables = Vec::new();
    for t in sh.tables.clone() {
        if range_has_data(sh, t.range) {
            keep_tables.push(t);
        } else {
            changed = true;
        }
    }
    if changed {
        sh.tables = keep_tables;
    }

    let mut keep_charts = Vec::new();
    for ch in sh.charts.clone() {
        let has_live_series = ch.series.iter().any(|s| {
            let (_, body) = split_sheet(&s.values);
            body.split([',', ';']).filter_map(|part| RangeRef::parse(part.trim())).any(|r| range_has_data(sh, r))
        });
        if has_live_series {
            keep_charts.push(ch);
        } else {
            changed = true;
        }
    }
    sh.charts = keep_charts;

    if let Some(sel_id) = app.selected_chart
        && !sh.charts.iter().any(|c| c.id == sel_id)
    {
        app.selected_chart = None;
    }

    changed
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColKind {
    Id,
    Category,
    Numeric,
    Total,
    Rate,
    Empty,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct DetectedCol {
    col: u32,
    letter: String,
    header: String,
    kind: ColKind,
    header_to_end_range: String,
    data_only_range: String,
    avg_abs: f64,
    sample_values: Vec<String>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct DetectedTable {
    name: String,
    full_range: RangeRef,
    header_row: Option<u32>,
    data_start_row: u32,
    data_end_row: u32,
    totals_row: Option<u32>,
    columns: Vec<DetectedCol>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct ComparisonSpec {
    chart_type: String,
    subtype: String,
    cat_col: u32,
    val_cols: Vec<u32>,
    multi_range: String,
    title: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct SheetSnapshot {
    wb_title: String,
    active_sheet: String,
    all_sheets: Vec<Value>,
    selection: String,
    active_cell: String,
    used_range: Option<RangeRef>,
    tables: Vec<DetectedTable>,
    existing_charts: Vec<Value>,
    markdown_grid: String,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
struct InsertedChartRecord {
    chart_type: String,
    cat_col: u32,
    val_cols: Vec<u32>,
    anchor: String,
}

#[cfg(not(target_arch = "wasm32"))]
struct TurnContext {
    user_prompt: String,
    charts_inserted: Vec<InsertedChartRecord>,
    cleared_existing_charts: bool,
    tool_success_count: usize,
    tool_errors: Vec<String>,
    /// Full command-surface access for this turn: run exactly the command the model asked for and
    /// skip the legacy single-purpose shortcuts (raw chart inserts, automatic sheet wipes, post-turn
    /// re-formatting).
    full_tool_access: bool,
    /// Whether the `ui_control` tool may reach UI-only actions this turn.
    allow_ui_tools: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl TurnContext {
    fn new(user_prompt: &str, full_tool_access: bool, allow_ui_tools: bool) -> Self {
        Self {
            user_prompt: user_prompt.to_string(),
            charts_inserted: Vec::new(),
            cleared_existing_charts: false,
            tool_success_count: 0,
            tool_errors: Vec::new(),
            full_tool_access,
            allow_ui_tools,
        }
    }
}

/// Inspect the live workbook and active sheet to detect all formal Excel tables AND grid cell tables,
/// classify every column (`Category`, `Numeric`, `Total`, `Rate`, `Id`), and build a coordinate-labelled grid table.
#[cfg(not(target_arch = "wasm32"))]
fn detect_sheet_tables(control_tx: &Option<Sender<ControlRequest>>, ctx: &egui::Context) -> Option<SheetSnapshot> {
    let doc_info = call_control(control_tx, ctx, "document.inspect", json!({})).ok()?;
    let wb_title = doc_info.get("title").and_then(Value::as_str).unwrap_or("Workbook").to_string();
    let active_idx = doc_info.get("activeSheet").and_then(Value::as_u64).unwrap_or(0) as usize;
    let selection = doc_info.get("selection").and_then(Value::as_str).unwrap_or("A1").to_string();
    let active_cell = doc_info.get("activeCell").and_then(Value::as_str).unwrap_or("A1").to_string();
    let all_sheets = doc_info.get("sheets").and_then(Value::as_array).cloned().unwrap_or_default();
    let active_sh_meta = all_sheets.get(active_idx).cloned().unwrap_or_else(|| json!({}));
    let active_sheet = active_sh_meta.get("name").and_then(Value::as_str).unwrap_or("Sheet1").to_string();
    let existing_charts = active_sh_meta.get("charts").and_then(Value::as_array).cloned().unwrap_or_default();

    let raw_read = call_control(control_tx, ctx, "sheet.read", json!({})).ok()?;
    let fmt_read = call_control(control_tx, ctx, "sheet.read", json!({"formatted": true})).ok();

    let used_range_str = raw_read.get("range").and_then(Value::as_str);
    let Some(used_str) = used_range_str else {
        return Some(SheetSnapshot {
            wb_title,
            active_sheet,
            all_sheets,
            selection,
            active_cell,
            used_range: None,
            tables: Vec::new(),
            existing_charts,
            markdown_grid: "(The sheet is empty — no cells have data yet)".into(),
        });
    };
    let used_range = RangeRef::parse(used_str)?;
    let raw_rows = raw_read.get("values").and_then(Value::as_array).cloned().unwrap_or_default();
    let fmt_rows = fmt_read.as_ref().and_then(|v| v.get("values")).and_then(Value::as_array).cloned().unwrap_or_default();

    let cell_raw = |row: u32, col: u32| -> Value {
        if row < used_range.start.row || row > used_range.end.row || col < used_range.start.col || col > used_range.end.col {
            return Value::Null;
        }
        let r_idx = (row - used_range.start.row) as usize;
        let c_idx = (col - used_range.start.col) as usize;
        raw_rows.get(r_idx).and_then(Value::as_array).and_then(|r| r.get(c_idx)).cloned().unwrap_or(Value::Null)
    };

    let cell_fmt = |row: u32, col: u32| -> String {
        if row < used_range.start.row || row > used_range.end.row || col < used_range.start.col || col > used_range.end.col {
            return String::new();
        }
        let r_idx = (row - used_range.start.row) as usize;
        let c_idx = (col - used_range.start.col) as usize;
        if let Some(s) = fmt_rows.get(r_idx).and_then(Value::as_array).and_then(|r| r.get(c_idx)).and_then(Value::as_str) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
        match cell_raw(row, col) {
            Value::Null => String::new(),
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            other => other.to_string(),
        }
    };

    // 1. Collect formal tables from `document.inspect`
    let mut table_specs: Vec<(String, RangeRef, bool)> = Vec::new();
    if let Some(formal_tables) = active_sh_meta.get("tables").and_then(Value::as_array) {
        for ft in formal_tables {
            let name = ft.get("name").and_then(Value::as_str).unwrap_or("Table").to_string();
            let totals = ft.get("totals").and_then(Value::as_bool).unwrap_or(false);
            if let Some(r_str) = ft.get("range").and_then(Value::as_str)
                && let Some(r) = RangeRef::parse(split_sheet(r_str).1)
            {
                table_specs.push((name, r, totals));
            }
        }
    }

    // 2. Also auto-detect plain grid cell tables (contiguous blocks of rows having >= 2 non-empty cells)
    // outside formal tables, so tables created via `write_range` without `create_table` are 100% identified!
    let formal_ranges: Vec<RangeRef> = table_specs.iter().map(|(_, r, _)| *r).collect();
    let is_row_in_formal = |row: u32| -> bool { formal_ranges.iter().any(|r| row >= r.start.row && row <= r.end.row) };

    let mut current_block: Option<(u32, u32, u32, u32)> = None;
    let mut auto_idx = 1usize;
    for row in used_range.start.row..=used_range.end.row {
        if is_row_in_formal(row) {
            if let Some((r0, r1, c0, c1)) = current_block.take()
                && r1 > r0
            {
                table_specs.push((format!("GridTable{auto_idx}"), RangeRef::new(CellRef::new(r0, c0), CellRef::new(r1, c1)), false));
                auto_idx += 1;
            }
            continue;
        }
        let mut non_empty_cols = Vec::new();
        for col in used_range.start.col..=used_range.end.col {
            if !cell_raw(row, col).is_null() && !cell_fmt(row, col).is_empty() {
                non_empty_cols.push(col);
            }
        }
        if non_empty_cols.len() >= 2 {
            let row_c0 = non_empty_cols.first().copied().unwrap_or(used_range.start.col);
            let row_c1 = non_empty_cols.last().copied().unwrap_or(used_range.end.col);
            match &mut current_block {
                Some((_, r1, c0, c1)) => {
                    *r1 = row;
                    *c0 = (*c0).min(row_c0);
                    *c1 = (*c1).max(row_c1);
                }
                None => {
                    current_block = Some((row, row, row_c0, row_c1));
                }
            }
        } else if let Some((r0, r1, c0, c1)) = current_block.take()
            && r1 > r0
        {
            table_specs.push((format!("GridTable{auto_idx}"), RangeRef::new(CellRef::new(r0, c0), CellRef::new(r1, c1)), false));
            auto_idx += 1;
        }
    }
    if let Some((r0, r1, c0, c1)) = current_block.take()
        && r1 > r0
    {
        table_specs.push((format!("GridTable{auto_idx}"), RangeRef::new(CellRef::new(r0, c0), CellRef::new(r1, c1)), false));
    }

    if table_specs.is_empty() {
        table_specs.push(("GridData".to_string(), used_range, false));
    }

    let mut tables = Vec::new();
    for (name, r, formal_totals) in table_specs {
        let mut text_headers = 0usize;
        let mut non_empty_top = 0usize;
        for col in r.start.col..=r.end.col {
            match cell_raw(r.start.row, col) {
                Value::String(s) if !s.trim().is_empty() => {
                    text_headers += 1;
                    non_empty_top += 1;
                }
                Value::Null => {}
                _ => non_empty_top += 1,
            }
        }
        let has_header = r.height() >= 2 && text_headers > 0 && text_headers * 2 >= non_empty_top;
        let header_row = has_header.then_some(r.start.row);
        let data_start_row = if has_header { r.start.row + 1 } else { r.start.row };

        // Check if bottom row is a Total / Jumlah / Rata-rata summary row
        let mut is_bottom_summary = formal_totals;
        if !is_bottom_summary && r.end.row > data_start_row {
            for col in r.start.col..=r.end.col {
                if let Value::String(s) = cell_raw(r.end.row, col) {
                    let low = s.trim().to_ascii_lowercase();
                    if low == "total"
                        || low == "jumlah"
                        || low == "subtotal"
                        || low == "grand total"
                        || low == "rata-rata"
                        || low == "rata rata"
                        || low == "average"
                        || low.starts_with("total ")
                        || low.starts_with("jumlah ")
                    {
                        is_bottom_summary = true;
                        break;
                    }
                }
            }
        }
        let totals_row = (is_bottom_summary && r.end.row > data_start_row).then_some(r.end.row);
        let data_end_row = if totals_row.is_some() { r.end.row - 1 } else { r.end.row };

        // First pass over columns to gather numbers/texts per column
        struct RawColStats {
            col: u32,
            letter: String,
            header: String,
            nums: Vec<f64>,
            texts: Vec<String>,
            samples: Vec<String>,
            has_pct_fmt: bool,
        }
        let mut raw_cols = Vec::new();
        for col in r.start.col..=r.end.col {
            let letter = col_to_letters(col);
            let header = if let Some(hr) = header_row {
                let h = cell_fmt(hr, col);
                if h.is_empty() { format!("Column {letter}") } else { h }
            } else {
                format!("Column {letter}")
            };
            let mut nums = Vec::new();
            let mut texts = Vec::new();
            let mut samples = Vec::new();
            let mut has_pct_fmt = false;
            for row in data_start_row..=data_end_row {
                let fmt_s = cell_fmt(row, col);
                if fmt_s.ends_with('%') {
                    has_pct_fmt = true;
                }
                if !fmt_s.is_empty() && samples.len() < 8 {
                    samples.push(fmt_s);
                }
                match cell_raw(row, col) {
                    Value::Number(n) => {
                        if let Some(f) = n.as_f64() {
                            nums.push(f);
                        }
                    }
                    Value::String(s) => {
                        let t = s.trim();
                        if !t.is_empty() {
                            if let Ok(parsed_n) = t.trim_end_matches('%').replace(',', "").parse::<f64>() {
                                nums.push(parsed_n);
                            } else {
                                texts.push(t.to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            raw_cols.push(RawColStats { col, letter, header, nums, texts, samples, has_pct_fmt });
        }

        let any_text_col = raw_cols.iter().any(|c| !c.texts.is_empty() && c.nums.is_empty());
        let num_col_count = raw_cols.iter().filter(|c| !c.nums.is_empty()).count();
        let max_col_avg = raw_cols
            .iter()
            .filter_map(|c| if c.nums.is_empty() { None } else { Some(c.nums.iter().map(|v| v.abs()).sum::<f64>() / c.nums.len() as f64) })
            .fold(0.0_f64, f64::max);

        let mut columns = Vec::new();
        for rc in raw_cols {
            let h_low = rc.header.trim().to_ascii_lowercase();
            let avg_abs = if rc.nums.is_empty() { 0.0 } else { rc.nums.iter().map(|v| v.abs()).sum::<f64>() / rc.nums.len() as f64 };
            let is_sequential_int = rc.nums.len() >= 2 && rc.nums.iter().enumerate().all(|(i, v)| (*v - (i as f64 + 1.0)).abs() < 1e-6);

            let kind = if rc.nums.is_empty() && rc.texts.is_empty() {
                ColKind::Empty
            } else if !rc.texts.is_empty() && rc.nums.is_empty() {
                ColKind::Category
            } else if !rc.nums.is_empty()
                && (matches!(h_low.as_str(), "no" | "no." | "id" | "#" | "nomor" | "urut" | "index" | "kode")
                    || (is_sequential_int && any_text_col && num_col_count >= 2))
            {
                ColKind::Id
            } else if !rc.nums.is_empty()
                && num_col_count >= 2
                && (h_low == "total" || h_low == "jumlah" || h_low == "subtotal" || h_low == "grand total" || h_low.starts_with("total "))
            {
                ColKind::Total
            } else if !rc.nums.is_empty()
                && (rc.has_pct_fmt
                    || h_low.contains('%')
                    || h_low.contains("growth")
                    || h_low.contains("pertumbuhan")
                    || h_low.contains("margin")
                    || h_low.contains("share")
                    || h_low.contains("porsi")
                    || h_low.contains("rasio")
                    || (avg_abs <= 1.5 && max_col_avg >= 20.0))
            {
                ColKind::Rate
            } else if !rc.nums.is_empty() {
                ColKind::Numeric
            } else {
                ColKind::Category
            };

            let top_r = header_row.unwrap_or(data_start_row);
            let header_to_end_range = format!("{}{}:{}{}", rc.letter, top_r + 1, rc.letter, data_end_row + 1);
            let data_only_range = format!("{}{}:{}{}", rc.letter, data_start_row + 1, rc.letter, data_end_row + 1);

            columns.push(DetectedCol {
                col: rc.col,
                letter: rc.letter,
                header: rc.header,
                kind,
                header_to_end_range,
                data_only_range,
                avg_abs,
                sample_values: rc.samples,
            });
        }

        tables.push(DetectedTable { name, full_range: r, header_row, data_start_row, data_end_row, totals_row, columns });
    }

    // Prioritize tables that have both category & numeric columns and more cells first
    tables.sort_by_key(|t| {
        let has_num = t.columns.iter().any(|c| matches!(c.kind, ColKind::Numeric | ColKind::Total | ColKind::Rate));
        let has_cat = t.columns.iter().any(|c| c.kind == ColKind::Category);
        (std::cmp::Reverse(has_num && has_cat), std::cmp::Reverse(has_num), std::cmp::Reverse(t.full_range.count()))
    });

    // Build Markdown Grid Table of `used_range` (up to 45 rows x 14 cols) with exact Column Letters and Row Numbers
    let mut md = String::new();
    let max_r = used_range.end.row.min(used_range.start.row + 44);
    let max_c = used_range.end.col.min(used_range.start.col + 13);
    md.push_str("| Row |");
    for col in used_range.start.col..=max_c {
        md.push_str(&format!(" {} |", col_to_letters(col)));
    }
    md.push('\n');
    md.push_str("|---|");
    for _ in used_range.start.col..=max_c {
        md.push_str("---|");
    }
    md.push('\n');
    for row in used_range.start.row..=max_r {
        let mut row_has_val = false;
        for col in used_range.start.col..=max_c {
            if !cell_fmt(row, col).is_empty() {
                row_has_val = true;
                break;
            }
        }
        if !row_has_val {
            continue;
        }
        md.push_str(&format!("| **{}** |", row + 1));
        for col in used_range.start.col..=max_c {
            let val = cell_fmt(row, col).replace('|', "/");
            md.push_str(&format!(" {val} |"));
        }
        md.push('\n');
    }

    Some(SheetSnapshot {
        wb_title,
        active_sheet,
        all_sheets,
        selection,
        active_cell,
        used_range: Some(used_range),
        tables,
        existing_charts,
        markdown_grid: md,
    })
}

/// Build up to 4 distinct, data-integrated comparison chart specifications for a `DetectedTable`.
#[cfg(not(target_arch = "wasm32"))]
fn build_comparison_visual_plan(table: &DetectedTable) -> Vec<ComparisonSpec> {
    let cat_cols: Vec<&DetectedCol> = table.columns.iter().filter(|c| c.kind == ColKind::Category).collect();
    let primary_cat = cat_cols.first().copied().or_else(|| table.columns.first());
    let Some(cat) = primary_cat else {
        return Vec::new();
    };

    let detail_cols: Vec<&DetectedCol> = table.columns.iter().filter(|c| c.kind == ColKind::Numeric && c.col != cat.col).collect();
    let total_cols: Vec<&DetectedCol> = table.columns.iter().filter(|c| c.kind == ColKind::Total && c.col != cat.col).collect();
    let rate_cols: Vec<&DetectedCol> = table.columns.iter().filter(|c| c.kind == ColKind::Rate && c.col != cat.col).collect();

    let mut all_metrics: Vec<&DetectedCol> = Vec::new();
    all_metrics.extend(detail_cols.iter().copied());
    all_metrics.extend(total_cols.iter().copied());
    all_metrics.extend(rate_cols.iter().copied());
    if all_metrics.is_empty() {
        // Fallback if only ID or other columns exist
        for c in &table.columns {
            if c.col != cat.col && c.kind != ColKind::Empty {
                all_metrics.push(c);
            }
        }
    }
    if all_metrics.is_empty() {
        return Vec::new();
    }

    let make_multi_range = |c_col: &DetectedCol, v_cols: &[&DetectedCol]| -> String {
        let mut parts = vec![c_col.header_to_end_range.clone()];
        for vc in v_cols {
            parts.push(vc.header_to_end_range.clone());
        }
        parts.join(",")
    };
    let join_headers = |v_cols: &[&DetectedCol]| -> String { v_cols.iter().map(|c| c.header.as_str()).collect::<Vec<_>>().join(", ") };

    // Check if `detail_cols` have compatible scale (max/min avg ratio <= 12x) so they look good on one Y-axis
    let detail_compatible = if detail_cols.len() >= 2 {
        let min_a = detail_cols.iter().map(|c| c.avg_abs.max(1e-6)).fold(f64::INFINITY, f64::min);
        let max_a = detail_cols.iter().map(|c| c.avg_abs).fold(0.0_f64, f64::max);
        max_a / min_a <= 12.0
    } else {
        false
    };

    let mut specs = Vec::new();

    if detail_cols.len() >= 2 && detail_compatible {
        // Spec 1: Clustered Column comparing all compatible detail columns (e.g. Q1, Q2, Q3, Q4)
        specs.push(ComparisonSpec {
            chart_type: "column".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: detail_cols.iter().map(|c| c.col).collect(),
            multi_range: make_multi_range(cat, &detail_cols),
            title: format!("Perbandingan {} per {}", join_headers(&detail_cols), cat.header),
        });

        // Spec 2: Pie / Doughnut of Total column (or last detail column)
        let pie_col = total_cols.first().copied().or_else(|| detail_cols.last().copied()).unwrap_or(detail_cols[0]);
        specs.push(ComparisonSpec {
            chart_type: "pie".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![pie_col.col],
            multi_range: make_multi_range(cat, &[pie_col]),
            title: format!("Proporsi {} per {}", pie_col.header, cat.header),
        });

        // Spec 3: Horizontal Bar of Rate/Growth column (or Line with Markers of first vs last detail column)
        if let Some(&rate_col) = rate_cols.first() {
            specs.push(ComparisonSpec {
                chart_type: "bar".into(),
                subtype: String::new(),
                cat_col: cat.col,
                val_cols: vec![rate_col.col],
                multi_range: make_multi_range(cat, &[rate_col]),
                title: format!("Perbandingan {} per {}", rate_col.header, cat.header),
            });
        } else {
            let first_last = if detail_cols.len() >= 3 { vec![detail_cols[0], detail_cols[detail_cols.len() - 1]] } else { vec![detail_cols[0]] };
            specs.push(ComparisonSpec {
                chart_type: "bar".into(),
                subtype: String::new(),
                cat_col: cat.col,
                val_cols: first_last.iter().map(|c| c.col).collect(),
                multi_range: make_multi_range(cat, &first_last),
                title: format!("Perbandingan {} per {}", join_headers(&first_last), cat.header),
            });
        }

        // Spec 4: Line with markers
        specs.push(ComparisonSpec {
            chart_type: "line".into(),
            subtype: "markers".into(),
            cat_col: cat.col,
            val_cols: detail_cols.iter().map(|c| c.col).collect(),
            multi_range: make_multi_range(cat, &detail_cols),
            title: format!("Tren {} per {}", join_headers(&detail_cols), cat.header),
        });
    } else if all_metrics.len() >= 2 {
        // Multiple numeric metrics with different scales (e.g. Harga, Terjual, Total) -> plot each metric in its own distinct visual!
        let m0 = all_metrics[0];
        let m1 = all_metrics.get(1).copied().unwrap_or(m0);
        let m2 = all_metrics.get(2).copied().or_else(|| total_cols.first().copied()).unwrap_or(m1);

        specs.push(ComparisonSpec {
            chart_type: "column".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![m0.col],
            multi_range: make_multi_range(cat, &[m0]),
            title: format!("Perbandingan {} per {}", m0.header, cat.header),
        });
        specs.push(ComparisonSpec {
            chart_type: "doughnut".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![m2.col],
            multi_range: make_multi_range(cat, &[m2]),
            title: format!("Proporsi {} per {}", m2.header, cat.header),
        });
        specs.push(ComparisonSpec {
            chart_type: "bar".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![m1.col],
            multi_range: make_multi_range(cat, &[m1]),
            title: format!("Perbandingan {} per {}", m1.header, cat.header),
        });
        specs.push(ComparisonSpec {
            chart_type: "line".into(),
            subtype: "markers".into(),
            cat_col: cat.col,
            val_cols: vec![m0.col],
            multi_range: make_multi_range(cat, &[m0]),
            title: format!("Tren {} per {}", m0.header, cat.header),
        });
    } else {
        // Single numeric column (e.g. Nama | Kategori | Nilai) -> 3 distinct chart types bound cleanly to that numeric column!
        let m = all_metrics[0];
        specs.push(ComparisonSpec {
            chart_type: "column".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![m.col],
            multi_range: make_multi_range(cat, &[m]),
            title: format!("Perbandingan {} per {}", m.header, cat.header),
        });
        specs.push(ComparisonSpec {
            chart_type: "doughnut".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![m.col],
            multi_range: make_multi_range(cat, &[m]),
            title: format!("Proporsi Kontribusi {} per {}", m.header, cat.header),
        });
        specs.push(ComparisonSpec {
            chart_type: "bar".into(),
            subtype: String::new(),
            cat_col: cat.col,
            val_cols: vec![m.col],
            multi_range: make_multi_range(cat, &[m]),
            title: format!("Ranking {} per {}", m.header, cat.header),
        });
        specs.push(ComparisonSpec {
            chart_type: "line".into(),
            subtype: "markers".into(),
            cat_col: cat.col,
            val_cols: vec![m.col],
            multi_range: make_multi_range(cat, &[m]),
            title: format!("Tren {} per {}", m.header, cat.header),
        });
    }

    specs
}

/// Compute non-overlapping dashboard anchor cells (`at`) for multiple charts next to and below the table.
#[cfg(not(target_arch = "wasm32"))]
fn dashboard_chart_anchor(table: &DetectedTable, chart_index: usize) -> String {
    let right_col = table.full_range.end.col + 2;
    let right_letter = col_to_letters(right_col);
    let left_letter = col_to_letters(table.full_range.start.col);
    let below_row = table.full_range.end.row + 3;

    match chart_index {
        0 => format!("{right_letter}2"),
        1 => format!("{right_letter}15"),
        2 => {
            if below_row <= 14 {
                format!("{left_letter}{below_row}")
            } else {
                format!("{right_letter}28")
            }
        }
        n => {
            let row = 2 + (n as u32) * 13;
            format!("{right_letter}{row}")
        }
    }
}

/// Build a comprehensive, coordinate-labelled description of the active workbook, sheets, tables,
/// columns, and cell values so Cloud API, Local Ollama, and Terminal Agents all know the exact grid state!
#[cfg(not(target_arch = "wasm32"))]
fn build_live_workbook_context(control_tx: &Option<Sender<ControlRequest>>, ctx: &egui::Context) -> String {
    let Some(snap) = detect_sheet_tables(control_tx, ctx) else {
        return "=== LIVE WORKBOOK CONTEXT ===\n(Could not read workbook state)\n".into();
    };

    let mut out = String::new();
    out.push_str("=== LIVE GRID & TABLE CONTEXT ===\n");
    out.push_str(&format!(
        "Workbook: \"{}\" | Active Sheet: \"{}\" | Active Cell: {} | Selection: {}\n",
        snap.wb_title, snap.active_sheet, snap.active_cell, snap.selection
    ));

    if snap.all_sheets.len() > 1 {
        let names: Vec<String> = snap
            .all_sheets
            .iter()
            .map(|s| {
                let n = s.get("name").and_then(Value::as_str).unwrap_or("Sheet");
                let u = s.get("usedRange").and_then(Value::as_str).unwrap_or("empty");
                format!("{n} (used: {u})")
            })
            .collect();
        out.push_str(&format!("All Sheets: {}\n", names.join(", ")));
    }

    if let Some(ur) = snap.used_range {
        out.push_str(&format!("Active Sheet Used Range: {}\n\n", ur.a1()));
    } else {
        out.push_str("Active Sheet Used Range: EMPTY (No cells populated yet)\n");
        return out;
    }

    out.push_str("--- DETECTED TABLES & COLUMNS ON ACTIVE SHEET ---\n");
    for (ti, t) in snap.tables.iter().enumerate() {
        let hdr_str = t.header_row.map(|r| format!("Row {}", r + 1)).unwrap_or_else(|| "None".into());
        let tot_str = t.totals_row.map(|r| format!("Row {} (EXCLUDED from chart ranges)", r + 1)).unwrap_or_else(|| "None".into());
        out.push_str(&format!(
            "Table #{} \"{}\" | Full Range: {} | Header: {} | Data Rows: {}..{} | Totals Row: {}\n",
            ti + 1,
            t.name,
            t.full_range.a1(),
            hdr_str,
            t.data_start_row + 1,
            t.data_end_row + 1,
            tot_str
        ));
        for c in &t.columns {
            let role = match c.kind {
                ColKind::Category => "CATEGORY / TEXT LABEL",
                ColKind::Numeric => "NUMERIC METRIC",
                ColKind::Total => "NUMERIC TOTAL / SUMMARY",
                ColKind::Rate => "NUMERIC RATE / PERCENTAGE",
                ColKind::Id => "ID / ROW NUMBER (Do not chart as series)",
                ColKind::Empty => "EMPTY / SPARKLINE (Do not chart)",
            };
            let preview = c.sample_values.join(", ");
            out.push_str(&format!(
                "  • Col {} (Header: \"{}\") [{}] | Header+Data: {} | Data-Only: {} | Values: [{}]\n",
                c.letter, c.header, role, c.header_to_end_range, c.data_only_range, preview
            ));
        }

        let plan = build_comparison_visual_plan(t);
        if !plan.is_empty() {
            out.push_str("  Recommended Distinct Comparison Visuals (`insert_chart`) for this table:\n");
            for (vi, spec) in plan.iter().take(3).enumerate() {
                let at = dashboard_chart_anchor(t, vi);
                out.push_str(&format!(
                    "    Visual #{}: {{\"name\":\"insert_chart\",\"arguments\":{{\"range\":\"{}\",\"type\":\"{}\",\"title\":\"{}\",\"at\":\"{}\"}}}}\n",
                    vi + 1,
                    spec.multi_range,
                    spec.chart_type,
                    spec.title,
                    at
                ));
            }
        }
        out.push('\n');
    }

    if !snap.existing_charts.is_empty() {
        out.push_str(&format!(
            "Existing Charts on Sheet ({}): {}\n\n",
            snap.existing_charts.len(),
            serde_json::to_string(&snap.existing_charts).unwrap_or_default()
        ));
    }

    out.push_str("--- ACTIVE SHEET CELL VALUES (Markdown Grid by Row & Column Letter) ---\n");
    out.push_str(&snap.markdown_grid);
    out.push_str("=================================\n");
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn call_control(control_tx: &Option<Sender<ControlRequest>>, ctx: &egui::Context, method: &str, params: Value) -> Result<Value, String> {
    let Some(tx) = control_tx else {
        return Err("control channel not connected".into());
    };
    let (req, rx) = ControlRequest::new(method, params);
    let state = Arc::clone(&req.state);
    tx.send(req).map_err(|_| "UI control channel closed".to_string())?;
    ctx.request_repaint();
    let reply = rx.recv_timeout(std::time::Duration::from_secs(15)).map_err(|_| {
        ControlRequest::abandon(&state);
        "timed out waiting for UI thread".to_string()
    })?;
    if reply.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(reply.get("result").cloned().unwrap_or(Value::Null))
    } else {
        Err(reply.get("error").and_then(Value::as_str).unwrap_or("command failed").to_string())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize_chart_type(raw: &str) -> (String, String) {
    let clean = raw.trim().to_ascii_lowercase().replace(['-', '_', ' '], "");
    match clean.as_str() {
        "column" | "columnchart" | "clusteredcolumn" | "verticalbar" => ("column".into(), String::new()),
        "stackedcolumn" | "columnstacked" => ("column".into(), "stacked".into()),
        "stackedcolumn100" | "columnstacked100" => ("column".into(), "stacked100".into()),
        "bar" | "barchart" | "clusteredbar" | "horizontalbar" => ("bar".into(), String::new()),
        "stackedbar" | "barstacked" => ("bar".into(), "stacked".into()),
        "line" | "linechart" => ("line".into(), "markers".into()),
        "linemarkers" | "linewithmarkers" => ("line".into(), "markers".into()),
        "pie" | "piechart" => ("pie".into(), String::new()),
        "doughnut" | "donut" | "doughnutchart" | "donutchart" => ("doughnut".into(), String::new()),
        "area" | "areachart" => ("area".into(), String::new()),
        "scatter" | "scatterplot" | "xy" => ("scatter".into(), String::new()),
        "radar" | "radarchart" => ("radar".into(), String::new()),
        "waterfall" => ("waterfall".into(), String::new()),
        "funnel" => ("funnel".into(), String::new()),
        "treemap" => ("treemap".into(), String::new()),
        "sunburst" => ("sunburst".into(), String::new()),
        "histogram" => ("histogram".into(), String::new()),
        "boxwhisker" | "boxplot" => ("boxWhisker".into(), String::new()),
        "combo" => ("combo".into(), String::new()),
        _ => ("column".into(), String::new()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn prompt_wants_append_chart(prompt: &str) -> bool {
    let low = prompt.to_ascii_lowercase();
    low.contains("tambah") || low.contains("add another") || low.contains("lagi")
}

#[cfg(not(target_arch = "wasm32"))]
fn prompt_wants_merge_tables(prompt: &str) -> bool {
    let low = prompt.to_ascii_lowercase();
    low.contains("gabung") || low.contains("satukan") || low.contains("kombinasi") || low.contains("merge") || low.contains("combine")
}

#[cfg(not(target_arch = "wasm32"))]
fn prompt_wants_delete_all(prompt: &str) -> bool {
    let low = prompt.trim().to_ascii_lowercase();
    if matches!(
        low.as_str(),
        "hapus"
            | "hapus semua"
            | "hapus semuanya"
            | "hapus data"
            | "hapus semua data"
            | "bersihkan"
            | "bersihkan semua"
            | "clear"
            | "clear all"
            | "delete"
            | "delete all"
            | "reset"
            | "kosongkan"
    ) {
        return true;
    }
    let has_del = low.contains("hapus")
        || low.contains("bersihkan")
        || low.contains("kosongkan")
        || low.contains("clear")
        || low.contains("delete")
        || low.contains("reset");
    let mentions_all = low.contains("semua")
        || low.contains("seluruh")
        || low.contains("all")
        || low.contains("sheet")
        || (low.contains("data")
            && (low.contains("chart") || low.contains("visual") || low.contains("grafik") || low.contains("table") || low.contains("tabel")));
    has_del && mentions_all
}

#[cfg(not(target_arch = "wasm32"))]
fn prompt_wants_delete_charts(prompt: &str) -> bool {
    let low = prompt.trim().to_ascii_lowercase();
    let has_del = low.contains("hapus") || low.contains("hilangkan") || low.contains("clear") || low.contains("delete") || low.contains("remove");
    let has_chart = low.contains("chart") || low.contains("visual") || low.contains("grafik") || low.contains("diagram");
    has_del && has_chart && !prompt_wants_delete_all(prompt)
}

#[cfg(not(target_arch = "wasm32"))]
fn humanize_tool_start(name: &str, args: &Value) -> String {
    let r = args.get("range").or_else(|| args.get("cell")).and_then(Value::as_str).unwrap_or("");
    match name {
        "inspect_workbook" => "Inspecting the active sheet structure…".into(),
        "read_range" => {
            if r.is_empty() {
                "Reading the active sheet data…".into()
            } else {
                format!("Reading cell {r}…")
            }
        }
        "write_range" => {
            let rows = args.get("values").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
            if rows > 0 && !r.is_empty() {
                format!("Writing {rows} rows to {r}…")
            } else if !r.is_empty() {
                format!("Writing data to {r}…")
            } else {
                "Writing data to the table…".into()
            }
        }
        "set_cell" => {
            if !r.is_empty() {
                format!("Setting cell {r}…")
            } else {
                "Setting the cell value…".into()
            }
        }
        "format_range" => {
            if !r.is_empty() {
                format!("Formatting {r}…")
            } else {
                "Formatting the table…".into()
            }
        }
        "create_table" => {
            if !r.is_empty() {
                format!("Creating a table on {r}…")
            } else {
                "Creating the table…".into()
            }
        }
        "insert_chart" => {
            let title = args.get("title").and_then(Value::as_str).unwrap_or("");
            if !title.is_empty() { format!("Creating chart {title}…") } else { "Creating a comparison chart…".into() }
        }
        "clear_range" => "Clearing data, fills and chart widgets…".into(),
        "delete_chart" => "Deleting chart widgets…".into(),
        "evaluate_formula" => "Menghitung rumus…".into(),
        "list_commands" => {
            let search = args.get("search").and_then(Value::as_str).unwrap_or("");
            if search.is_empty() { "Mencari command GridCraft…".into() } else { format!("Mencari command \"{search}\"…") }
        }
        "describe_command" => {
            let cmd = args.get("command").and_then(Value::as_str).unwrap_or("command");
            format!("Reading command details: {cmd}…")
        }
        "execute_commands" => {
            let n = args.get("commands").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
            format!("Running {n} commands…")
        }
        "list_functions" => "Mencari fungsi worksheet…".into(),
        "get_cell" => {
            if r.is_empty() {
                "Reading the cell…".into()
            } else {
                format!("Reading cell {r}…")
            }
        }
        "select" => {
            if r.is_empty() {
                "Memilih range…".into()
            } else {
                format!("Memilih {r}…")
            }
        }
        "sort_range" => {
            if r.is_empty() {
                "Mengurutkan data…".into()
            } else {
                format!("Mengurutkan {r}…")
            }
        }
        "undo" => "Undo…".into(),
        "redo" => "Redo…".into(),
        "create_report" => "Building a clean report…".into(),
        "layout_sheet" => "Applying a readable layout…".into(),
        "ui_control" => {
            let method = args.get("method").and_then(Value::as_str).unwrap_or("ui");
            format!("Running UI action {method}…")
        }
        "execute_command" => {
            let cmd = args.get("command").and_then(Value::as_str).unwrap_or("command");
            format!("Running command {cmd}…")
        }
        _ => format!("Running {name}…"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn humanize_tool_done(name: &str, args: &Value, result: &Value) -> String {
    let r = args.get("range").or_else(|| args.get("cell")).or_else(|| result.get("range")).and_then(Value::as_str).unwrap_or("");
    match name {
        "inspect_workbook" => "Checked the active sheet structure".into(),
        "read_range" => {
            if !r.is_empty() {
                format!("Read data on {r}")
            } else {
                "Read the active sheet data".into()
            }
        }
        "write_range" => {
            let rows = args.get("values").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
            if rows > 0 && !r.is_empty() {
                format!("Wrote {rows} rows to {r}")
            } else if !r.is_empty() {
                format!("Wrote data to {r}")
            } else {
                "Wrote data to the table".into()
            }
        }
        "set_cell" => {
            if !r.is_empty() {
                format!("Updated cell {r}")
            } else {
                "Updated the cell".into()
            }
        }
        "format_range" => {
            if !r.is_empty() {
                format!("Formatted {r}")
            } else {
                "Formatted the table".into()
            }
        }
        "create_table" => {
            if !r.is_empty() {
                format!("Created a table on {r}")
            } else {
                "Created the table".into()
            }
        }
        "insert_chart" => {
            let ctype = result.get("type").or_else(|| args.get("type")).and_then(Value::as_str).unwrap_or("chart");
            let title = result.get("title").or_else(|| args.get("title")).and_then(Value::as_str).unwrap_or("Visual");
            let at = result.get("at").or_else(|| args.get("at")).and_then(Value::as_str).unwrap_or("");
            if !at.is_empty() { format!("Created {ctype} chart: {title} (at {at})") } else { format!("Created {ctype} chart: {title}") }
        }
        "clear_range" => "Cleared data, fills, tables and chart widgets".into(),
        "delete_chart" => "Deleted chart widgets".into(),
        "evaluate_formula" => "Formula evaluated".into(),
        "list_commands" => "Searched commands".into(),
        "describe_command" => "Read command details".into(),
        "execute_commands" => {
            let n = args.get("commands").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
            format!("{n} commands executed")
        }
        "list_functions" => "Read worksheet functions".into(),
        "get_cell" => {
            if r.is_empty() {
                "Read the cell".into()
            } else {
                format!("Read cell {r}")
            }
        }
        "select" => {
            if r.is_empty() {
                "Selection diperbarui".into()
            } else {
                format!("Selection diperbarui ke {r}")
            }
        }
        "sort_range" => {
            if r.is_empty() {
                "Sorted the data".into()
            } else {
                format!("Sorted {r}")
            }
        }
        "undo" => "Undone".into(),
        "redo" => "Redone".into(),
        "create_report" => "Built a clean report".into(),
        "layout_sheet" => "Layout applied".into(),
        "ui_control" => {
            let method = args.get("method").and_then(Value::as_str).unwrap_or("ui");
            format!("UI action {method} finished")
        }
        "execute_command" => {
            let cmd = args.get("command").and_then(Value::as_str).unwrap_or("command");
            format!("Command {cmd} finished")
        }
        _ => format!("{name} finished"),
    }
}

/// Detect if the user prompt explicitly asked to create N charts/visuals (e.g. "buatkan 3 visual perbandingan").
#[cfg(not(target_arch = "wasm32"))]
fn requested_chart_count(prompt: &str) -> Option<usize> {
    let low = prompt.to_ascii_lowercase();
    let has_create = low.contains("buat")
        || low.contains("bikin")
        || low.contains("tampilkan")
        || low.contains("create")
        || low.contains("generate")
        || low.contains("insert")
        || low.contains("make");
    let has_visual = low.contains("visual") || low.contains("chart") || low.contains("grafik") || low.contains("diagram");
    if !(has_create && has_visual) {
        return None;
    }

    let tokens: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()).collect();
    for (i, &tok) in tokens.iter().enumerate() {
        let next_is_visual = tokens.get(i + 1).is_some_and(|&nx| {
            nx.starts_with("visual")
                || nx.starts_with("chart")
                || nx.starts_with("grafik")
                || nx.starts_with("diagram")
                || nx.starts_with("perbandingan")
        });
        if next_is_visual {
            let n = match tok {
                "1" | "satu" | "one" => Some(1),
                "2" | "dua" | "two" => Some(2),
                "3" | "tiga" | "three" => Some(3),
                "4" | "empat" | "four" => Some(4),
                "5" | "lima" | "five" => Some(5),
                _ => None,
            };
            if let Some(count) = n {
                return Some(count);
            }
        }
    }
    Some(1)
}

/// Smart Chart Integrator & Multi-Visual Engine:
/// - Synchronizes merged/expanded table formatting with Table 1 first so the chart range covers all merged rows.
/// - Automatically binds charts to the active table's real category column and real numeric columns
///   (skipping ID columns, non-numeric text columns like `Kategori`, empty sparkline columns, and bottom `Total` summary rows).
/// - Guarantees that when multiple comparison visuals are created in a turn, each visual has a distinct chart type,
///   meaningful comparative data series, visible `dataLabels`, and non-overlapping dashboard coordinates!
#[cfg(not(target_arch = "wasm32"))]
fn smart_insert_chart(
    control_tx: &Option<Sender<ControlRequest>>,
    ctx: &egui::Context,
    args: &Value,
    turn_ctx: &mut TurnContext,
) -> Result<Value, String> {
    let exec = |cmd: &str, p: Value| call_control(control_tx, ctx, "engine.execute", json!({"command": cmd, "params": p}));

    // Synchronize merged/expanded table formatting with Table 1 BEFORE reading the table for charting!
    let _ = call_control(control_tx, ctx, "ai.sync_table_format", json!({"merge_requested": prompt_wants_merge_tables(&turn_ctx.user_prompt)}));

    // 1. Clear leftover charts on the sheet before inserting the first chart of a fresh "buatkan ... visual" request
    if !turn_ctx.cleared_existing_charts && !prompt_wants_append_chart(&turn_ctx.user_prompt) {
        turn_ctx.cleared_existing_charts = true;
        if let Some(snap) = detect_sheet_tables(control_tx, ctx) {
            for ch in &snap.existing_charts {
                if let Some(id) = ch.get("id").and_then(Value::as_u64) {
                    let _ = exec("chart.delete", json!({"chart": id}));
                }
            }
        }
    }

    let snap = detect_sheet_tables(control_tx, ctx).ok_or_else(|| "Failed to read sheet data to create chart.".to_string())?;
    let Some(table) = snap.tables.first() else {
        return Err("No table or data found on the active sheet to visualize.".into());
    };

    let plan = build_comparison_visual_plan(table);
    if plan.is_empty() {
        return Err("The table on the active sheet has no numeric columns to visualize.".into());
    }

    let empty_map = Map::new();
    let a = args.as_object().unwrap_or(&empty_map);

    let raw_type = a
        .get("type")
        .or_else(|| a.get("chart_type"))
        .or_else(|| a.get("kind"))
        .or_else(|| a.get("chartType"))
        .and_then(Value::as_str)
        .unwrap_or("column");
    let (mut req_type, mut req_sub) = normalize_chart_type(raw_type);
    if let Some(s) = a.get("subtype").and_then(Value::as_str)
        && !s.trim().is_empty()
    {
        req_sub = s.trim().to_string();
    }

    let raw_title = a.get("title").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("Chart Title"));

    // Determine which columns of `table` the AI requested (via `range`, `columns`, or `title`)
    let raw_range = a
        .get("range")
        .or_else(|| a.get("data_range"))
        .or_else(|| a.get("source"))
        .or_else(|| a.get("dataRange"))
        .and_then(Value::as_str)
        .unwrap_or("");

    let cat_col_default = table
        .columns
        .iter()
        .find(|c| c.kind == ColKind::Category)
        .or_else(|| table.columns.first())
        .map(|c| c.col)
        .unwrap_or(table.full_range.start.col);

    let mut chosen_cat_col = cat_col_default;
    let mut chosen_val_cols: Vec<u32> = Vec::new();

    if !raw_range.trim().is_empty() {
        let mut parsed_areas = Vec::new();
        for part in raw_range.split([',', ';']) {
            let (_, body) = split_sheet(part.trim());
            if let Some(r) = RangeRef::parse(body.trim()) {
                parsed_areas.push(r);
            }
        }
        if parsed_areas.len() > 1 {
            // Multi-area range: first area is category if it's a Category column, rest are value columns
            let first_c = parsed_areas[0].start.col;
            if table.columns.iter().any(|c| c.col == first_c && c.kind == ColKind::Category) {
                chosen_cat_col = first_c;
            }
            for area in parsed_areas.iter().skip(1) {
                for col in area.start.col..=area.end.col {
                    if let Some(tc) = table.columns.iter().find(|c| c.col == col)
                        && matches!(tc.kind, ColKind::Numeric | ColKind::Total | ColKind::Rate)
                        && !chosen_val_cols.contains(&col)
                    {
                        chosen_val_cols.push(col);
                    }
                }
            }
        } else if let Some(single) = parsed_areas.first() {
            // Single contiguous range: if it spans a narrow sub-selection of numeric columns, keep those
            let spans_whole_table = single.width() >= table.full_range.width().saturating_sub(1);
            if !spans_whole_table {
                for col in single.start.col..=single.end.col {
                    if let Some(tc) = table.columns.iter().find(|c| c.col == col)
                        && matches!(tc.kind, ColKind::Numeric | ColKind::Total | ColKind::Rate)
                        && !chosen_val_cols.contains(&col)
                    {
                        chosen_val_cols.push(col);
                    }
                }
            }
        }
    }

    // If `chosen_val_cols` is still empty or spans incompatible columns, check `title` for column header matches
    if chosen_val_cols.is_empty()
        && let Some(t_str) = raw_title
    {
        let t_low = t_str.to_ascii_lowercase();
        for tc in &table.columns {
            if matches!(tc.kind, ColKind::Numeric | ColKind::Total | ColKind::Rate) {
                let h_low = tc.header.to_ascii_lowercase();
                if !h_low.is_empty() && t_low.contains(&h_low) {
                    chosen_val_cols.push(tc.col);
                }
            }
        }
    }

    // Single-series charts (`pie`, `doughnut`, `funnel`, `treemap`, `waterfall`) can only plot 1 numeric column!
    let is_single_series_kind = matches!(req_type.as_str(), "pie" | "doughnut" | "funnel" | "treemap" | "sunburst" | "waterfall");
    if is_single_series_kind && chosen_val_cols.len() > 1 {
        let preferred = chosen_val_cols
            .iter()
            .copied()
            .find(|&c| table.columns.iter().any(|tc| tc.col == c && tc.kind == ColKind::Total))
            .or_else(|| chosen_val_cols.last().copied());
        chosen_val_cols = preferred.into_iter().collect();
    }

    // Check if this chart would duplicate a visual already inserted in this turn (same chart_type OR same (chart_type, val_cols))
    let chart_turn_idx = turn_ctx.charts_inserted.len();
    let type_already_used = turn_ctx.charts_inserted.iter().any(|prev| prev.chart_type == req_type);
    let exact_dup =
        turn_ctx.charts_inserted.iter().any(|prev| prev.chart_type == req_type && prev.cat_col == chosen_cat_col && prev.val_cols == chosen_val_cols);

    let mut final_title = raw_title.map(str::to_string);

    if chosen_val_cols.is_empty() || exact_dup || (chart_turn_idx > 0 && type_already_used) {
        // Pick the first spec from `plan` whose `chart_type` (or `(chart_type, val_cols)`) hasn't been used yet!
        let fallback_spec = plan
            .iter()
            .find(|sp| !turn_ctx.charts_inserted.iter().any(|prev| prev.chart_type == sp.chart_type))
            .or_else(|| plan.get(chart_turn_idx % plan.len()))
            .unwrap_or(&plan[0]);

        if chosen_val_cols.is_empty() || exact_dup {
            chosen_cat_col = fallback_spec.cat_col;
            chosen_val_cols = fallback_spec.val_cols.clone();
            final_title = Some(fallback_spec.title.clone());
        }
        if type_already_used || exact_dup {
            req_type = fallback_spec.chart_type.clone();
            req_sub = fallback_spec.subtype.clone();
            if final_title.is_none() || exact_dup {
                final_title = Some(fallback_spec.title.clone());
            }
        }
    }

    // Re-check single-series constraint if `req_type` changed to `pie` / `doughnut`
    if matches!(req_type.as_str(), "pie" | "doughnut" | "funnel" | "treemap" | "sunburst" | "waterfall")
        && chosen_val_cols.len() > 1
        && let Some(last_c) = chosen_val_cols.last().copied()
    {
        chosen_val_cols = vec![last_c];
    }

    // Build the exact multi-area range: `{cat_col_range},{val_col1_range},...`
    let top_row = table.header_row.unwrap_or(table.data_start_row);
    let end_row = table.data_end_row;
    let cat_letter = col_to_letters(chosen_cat_col);
    let mut range_parts = vec![format!("{cat_letter}{}:{cat_letter}{}", top_row + 1, end_row + 1)];
    let mut val_headers = Vec::new();
    for &vc in &chosen_val_cols {
        let vl = col_to_letters(vc);
        range_parts.push(format!("{vl}{}:{vl}{}", top_row + 1, end_row + 1));
        if let Some(tc) = table.columns.iter().find(|c| c.col == vc) {
            val_headers.push(tc.header.clone());
        }
    }
    let multi_range_str = range_parts.join(",");

    let cat_header = table.columns.iter().find(|c| c.col == chosen_cat_col).map(|c| c.header.as_str()).unwrap_or("Kategori");

    let resolved_title = final_title.unwrap_or_else(|| {
        let prefix = match req_type.as_str() {
            "pie" | "doughnut" => "Proporsi",
            "line" | "area" => "Tren",
            "bar" => "Ranking",
            _ => "Perbandingan",
        };
        format!("{prefix} {} per {cat_header}", val_headers.join(", "))
    });

    // Pick a clean non-overlapping anchor cell (`at`)
    let raw_at = a.get("at").and_then(Value::as_str).map(str::trim).unwrap_or("");
    let at_already_used = !raw_at.is_empty() && turn_ctx.charts_inserted.iter().any(|prev| prev.anchor.eq_ignore_ascii_case(raw_at));
    let at_overlaps_table = !raw_at.is_empty() && CellRef::parse(raw_at).is_some_and(|c| table.full_range.contains(c));
    let final_anchor =
        if raw_at.is_empty() || at_already_used || at_overlaps_table { dashboard_chart_anchor(table, chart_turn_idx) } else { raw_at.to_string() };

    // Execute `insert.chart` with the exact multi-area range
    let mut insert_params = json!({
        "range": multi_range_str,
        "type": req_type,
        "title": resolved_title,
        "at": final_anchor,
        "width": 430,
        "height": 240
    });
    if !req_sub.is_empty() {
        insert_params["subtype"] = json!(req_sub);
    }

    let insert_res = match exec("insert.chart", insert_params) {
        Ok(v) => v,
        Err(_) => {
            // Fallback if `cat_col` had numeric category labels: insert on first value col, then attach categories via `chart.selectData`
            let vc = chosen_val_cols.first().copied().unwrap_or(chosen_cat_col);
            let vl = col_to_letters(vc);
            let val_range = format!("{vl}{}:{vl}{}", top_row + 1, end_row + 1);
            let v = exec(
                "insert.chart",
                json!({
                    "range": val_range,
                    "type": req_type,
                    "title": resolved_title,
                    "at": final_anchor,
                    "width": 430,
                    "height": 240
                }),
            )?;
            if let Some(cid) = v.get("chart").and_then(Value::as_u64) {
                let cat_data_range = format!("{cat_letter}{}:{cat_letter}{}", table.data_start_row + 1, end_row + 1);
                let _ = exec("chart.selectData", json!({"chart": cid, "categories": cat_data_range}));
            }
            v
        }
    };

    // Enable `dataLabels: true` and distinct visual style via `chart.set` so values are visibly integrated on the chart!
    if let Some(cid) = insert_res.get("chart").and_then(Value::as_u64) {
        let style_id = ((chart_turn_idx % 6) + 1) as u32;
        let _ = exec(
            "chart.set",
            json!({
                "chart": cid,
                "title": resolved_title,
                "at": final_anchor,
                "width": 430,
                "height": 240,
                "dataLabels": true,
                "style": style_id
            }),
        );
    }

    turn_ctx.charts_inserted.push(InsertedChartRecord {
        chart_type: req_type.clone(),
        cat_col: chosen_cat_col,
        val_cols: chosen_val_cols,
        anchor: final_anchor.clone(),
    });

    Ok(json!({
        "chart": insert_res.get("chart"),
        "type": req_type,
        "range": multi_range_str,
        "title": resolved_title,
        "at": final_anchor,
        "dataLabels": true
    }))
}

/// Post-turn consistency engine:
/// 1. If the user asked to delete/clear all (`"hapus"`, `"hapus semua"`, `"clear"`, `"reset"`),
///    executes `"ai.deep_clear"` so 100% of data, cell background colors, formal tables, sparklines,
///    conditional formats, and chart widgets are removed!
/// 2. If the user asked to delete charts (`"hapus chart"` / `"hapus visual"`), removes all chart widgets.
/// 3. Otherwise:
///    - Synchronizes merged/expanded table formatting with Table 1's blueprint (`"ai.sync_table_format"`).
///    - Cleans up any orphaned chart widgets or empty formal tables (`"ai.cleanup_orphaned_charts"`).
///    - Auto-completes any remaining requested comparison visuals (`requested_chart_count`).
#[cfg(not(target_arch = "wasm32"))]
fn ensure_post_turn_consistency(
    control_tx: &Option<Sender<ControlRequest>>,
    tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
    turn_ctx: &mut TurnContext,
) {
    if cancel.load(Ordering::SeqCst) {
        return;
    }

    if prompt_wants_delete_all(&turn_ctx.user_prompt) {
        let _ = call_control(control_tx, ctx, "ai.deep_clear", json!({"all": true}));
        return;
    }

    if prompt_wants_delete_charts(&turn_ctx.user_prompt) {
        if let Some(snap) = detect_sheet_tables(control_tx, ctx) {
            for ch in &snap.existing_charts {
                if let Some(id) = ch.get("id").and_then(Value::as_u64) {
                    let _ = call_control(control_tx, ctx, "engine.execute", json!({"command": "chart.delete", "params": {"chart": id}}));
                }
            }
        }
        let _ = call_control(control_tx, ctx, "ai.cleanup_orphaned_charts", json!({}));
        return;
    }

    // Synchronize merged/expanded table formatting with Table 1's blueprint
    let _ = call_control(control_tx, ctx, "ai.sync_table_format", json!({"merge_requested": prompt_wants_merge_tables(&turn_ctx.user_prompt)}));

    // Clean up any orphaned chart widgets or empty formal tables left behind after partial clears
    let _ = call_control(control_tx, ctx, "ai.cleanup_orphaned_charts", json!({}));

    let Some(target_count) = requested_chart_count(&turn_ctx.user_prompt) else {
        return;
    };
    while turn_ctx.charts_inserted.len() < target_count {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        let next_idx = turn_ctx.charts_inserted.len() + 1;
        let _ = tx.send(WorkerEvent::ToolStart { name: "insert_chart".into(), args: format!("Creating comparison chart #{next_idx}…") });
        ctx.request_repaint();
        match smart_insert_chart(control_tx, ctx, &json!({}), turn_ctx) {
            Ok(v) => {
                turn_ctx.tool_success_count += 1;
                let _ = tx.send(WorkerEvent::ToolDone {
                    name: "insert_chart".into(),
                    summary: humanize_tool_done("insert_chart", &json!({}), &v),
                    is_error: false,
                });
                ctx.request_repaint();
            }
            Err(e) => {
                turn_ctx.tool_errors.push(e.clone());
                let _ =
                    tx.send(WorkerEvent::ToolDone { name: "insert_chart".into(), summary: format!("Failed to create chart: {e}"), is_error: true });
                ctx.request_repaint();
                break;
            }
        }
    }
}

/// Read the live command catalog over the in-process control channel (every engine command plus the
/// desktop's UI-only commands). Returns an empty list when the UI thread cannot answer; callers then
/// fall back to the command id the model supplied.
#[cfg(not(target_arch = "wasm32"))]
fn fetch_command_catalog(control_tx: &Option<Sender<ControlRequest>>, ctx: &egui::Context) -> Vec<crate::ai_tools::CommandInfo> {
    match call_control(control_tx, ctx, "engine.commands", json!({})) {
        Ok(value) => crate::ai_tools::parse_catalog(&value),
        Err(_) => Vec::new(),
    }
}

/// Resolve the command a model asked for, turning unknown or known-unimplemented ids into messages
/// that tell it what to do next (list_commands, describe_command, nearest candidates).
#[cfg(not(target_arch = "wasm32"))]
fn resolve_requested_command(
    control_tx: &Option<Sender<ControlRequest>>,
    ctx: &egui::Context,
    raw: &str,
) -> Result<(String, Vec<crate::ai_tools::CommandInfo>), String> {
    let catalog = fetch_command_catalog(control_tx, ctx);
    if catalog.is_empty() {
        if raw.trim().is_empty() {
            return Err("missing `command`: pass a GridCraft command id, e.g. \"home.bold\"".into());
        }
        return Ok((raw.trim().to_string(), catalog));
    }
    if let Some(feature) = crate::ai_tools::known_unimplemented(raw) {
        return Err(format!(
            "`{raw}` ({feature}) is a known-unimplemented Excel catalog entry in GridCraft. Pick a supported alternative from list_commands."
        ));
    }
    let id = crate::ai_tools::resolve_command_id(&catalog, raw)?;
    Ok((id, catalog))
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_mcp_tool(
    control_tx: &Option<Sender<ControlRequest>>,
    ctx: &egui::Context,
    name: &str,
    args: &Value,
    turn_ctx: &mut TurnContext,
) -> Result<Value, String> {
    let empty = Map::new();
    let a = args.as_object().unwrap_or(&empty);
    let exec = |cmd: &str, p: Value| call_control(control_tx, ctx, "engine.execute", json!({"command": cmd, "params": p}));

    match name {
        "inspect_workbook" => call_control(control_tx, ctx, "document.inspect", json!({})),
        "read_range" => call_control(control_tx, ctx, "sheet.read", args.clone()),
        "write_range" => {
            // Check if the AI called `write_range` with all empty/null values to clear a range
            let all_values_blank = a.get("values").and_then(Value::as_array).is_some_and(|rows| {
                !rows.is_empty()
                    && rows
                        .iter()
                        .all(|r| r.as_array().is_some_and(|cols| cols.iter().all(|v| v.is_null() || v.as_str().is_some_and(|s| s.trim().is_empty()))))
            });
            if all_values_blank {
                let r_str = a.get("range").and_then(Value::as_str).unwrap_or("");
                if turn_ctx.full_tool_access {
                    // Blank values mean "clear these cells"; a plain write must not also wipe their
                    // formats, so use Clear Contents instead of the legacy deep clear.
                    return exec("edit.clearContents", json!({"range": r_str}));
                }
                return call_control(control_tx, ctx, "ai.deep_clear", json!({"range": r_str}));
            }
            let res = exec("range.setValues", args.clone())?;
            if !turn_ctx.full_tool_access {
                let _ = call_control(
                    control_tx,
                    ctx,
                    "ai.sync_table_format",
                    json!({"merge_requested": prompt_wants_merge_tables(&turn_ctx.user_prompt)}),
                );
            }
            Ok(res)
        }
        "set_cell" => {
            let is_blank = a.get("input").is_some_and(|v| v.is_null() || v.as_str().is_some_and(|s| s.trim().is_empty()));
            if is_blank {
                let c_str = a.get("cell").and_then(Value::as_str).unwrap_or("");
                if turn_ctx.full_tool_access {
                    return exec("edit.clearContents", json!({"range": c_str}));
                }
                return call_control(control_tx, ctx, "ai.deep_clear", json!({"range": c_str}));
            }
            exec("cell.set", args.clone())
        }
        "get_cell" => exec("cell.get", args.clone()),
        "create_table" => exec("insert.table", args.clone()),
        "insert_chart" => {
            if turn_ctx.full_tool_access {
                // Run the real command with the model's own type/anchor/title: no hidden chart
                // substitution, no deleting the sheet's existing charts.
                let p = crate::ai_tools::merge_command_params(a);
                exec("insert.chart", p).map_err(|e| format!("command `insert.chart` failed: {e}"))
            } else {
                smart_insert_chart(control_tx, ctx, args, turn_ctx)
            }
        }
        "clear_range" | "clear" | "clear_all" | "delete_all" => call_control(control_tx, ctx, "ai.deep_clear", args.clone()),
        "delete_chart" => {
            if let Some(cid) = a.get("chart").or_else(|| a.get("id")).and_then(Value::as_u64) {
                let _ = exec("chart.delete", json!({"chart": cid}));
            } else if let Some(snap) = detect_sheet_tables(control_tx, ctx) {
                for ch in &snap.existing_charts {
                    if let Some(id) = ch.get("id").and_then(Value::as_u64) {
                        let _ = exec("chart.delete", json!({"chart": id}));
                    }
                }
            }
            if !turn_ctx.full_tool_access {
                let _ = call_control(control_tx, ctx, "ai.cleanup_orphaned_charts", json!({}));
            }
            Ok(json!({"deleted": true}))
        }
        "evaluate_formula" => {
            let f = a.get("formula").and_then(Value::as_str).ok_or("missing `formula`")?;
            let formula = if f.starts_with('=') { f.to_string() } else { format!("={f}") };
            let mut p = json!({"formula": formula});
            if let Some(c) = a.get("cell") {
                p["cell"] = c.clone();
            }
            exec("formulas.evaluate", p)
        }
        "format_range" => {
            let range = a.get("range").and_then(Value::as_str).ok_or("missing `range`")?;
            let r = json!(range);
            let mut applied = Vec::new();
            if let Some(v) = a.get("bold").and_then(Value::as_bool) {
                exec("home.bold", json!({"range": r, "on": v}))?;
                applied.push("bold");
            }
            if let Some(v) = a.get("italic").and_then(Value::as_bool) {
                exec("home.italic", json!({"range": r, "on": v}))?;
                applied.push("italic");
            }
            if let Some(v) = a.get("underline").and_then(Value::as_bool) {
                exec("home.underline", json!({"range": r, "on": v}))?;
                applied.push("underline");
            }
            if let Some(v) = a.get("fontName").and_then(Value::as_str) {
                exec("home.fontName", json!({"range": r, "name": v}))?;
                applied.push("fontName");
            }
            if let Some(v) = a.get("fontSize").and_then(Value::as_f64) {
                exec("home.fontSize", json!({"range": r, "size": v}))?;
                applied.push("fontSize");
            }
            if let Some(v) = a.get("fontColor").and_then(Value::as_str) {
                exec("home.fontColor", json!({"range": r, "color": v}))?;
                applied.push("fontColor");
            }
            if let Some(v) = a.get("fillColor").and_then(Value::as_str) {
                exec("home.fillColor", json!({"range": r, "color": v}))?;
                applied.push("fillColor");
            }
            if let Some(v) = a.get("numberFormat").and_then(Value::as_str) {
                exec("home.numberFormat", json!({"range": r, "format": v}))?;
                applied.push("numberFormat");
            }
            if let Some(v) = a.get("horizontalAlign").and_then(Value::as_str) {
                let id = match v {
                    "left" => "home.alignLeft",
                    "center" => "home.alignCenter",
                    "right" => "home.alignRight",
                    _ => "home.justify",
                };
                exec(id, json!({"range": r}))?;
                applied.push("horizontalAlign");
            }
            if let Some(v) = a.get("wrapText").and_then(Value::as_bool) {
                exec("home.wrapText", json!({"range": r, "on": v}))?;
                applied.push("wrapText");
            }
            Ok(json!({"range": range, "applied": applied}))
        }
        "list_commands" => {
            let catalog = fetch_command_catalog(control_tx, ctx);
            if catalog.is_empty() {
                return call_control(control_tx, ctx, "engine.commands", args.clone());
            }
            let search = a.get("search").and_then(Value::as_str).unwrap_or("");
            let area = a.get("area").and_then(Value::as_str);
            let limit = a.get("limit").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(200).clamp(1, 500);
            let hits = crate::ai_tools::search_commands(&catalog, search, area, limit);
            let commands: Vec<Value> = hits.iter().map(|c| c.to_compact()).collect();
            Ok(json!({
                "count": commands.len(),
                "total": catalog.len(),
                "hint": "Run any id with execute_command {command, params}; describe_command shows its full params.",
                "commands": commands,
            }))
        }
        "describe_command" => {
            let raw = a.get("command").or_else(|| a.get("id")).and_then(Value::as_str).unwrap_or("");
            let (resolved, catalog) = resolve_requested_command(control_tx, ctx, raw)?;
            match catalog.iter().find(|c| c.id == resolved) {
                Some(info) => Ok(info.to_full()),
                None => Err(format!("command `{resolved}` is not in the live catalog")),
            }
        }
        "execute_command" => {
            let raw = a.get("command").or_else(|| a.get("id")).or_else(|| a.get("command_id")).and_then(Value::as_str).unwrap_or("");
            let (resolved, _catalog) = resolve_requested_command(control_tx, ctx, raw)?;
            let params = crate::ai_tools::merge_command_params(a);
            if (resolved == "insert.chart" || resolved == "insert.recommendedCharts") && !turn_ctx.full_tool_access {
                return smart_insert_chart(control_tx, ctx, &params, turn_ctx);
            }
            if !turn_ctx.full_tool_access
                && matches!(resolved.as_str(), "edit.clear" | "edit.clearAll" | "edit.clearContents" | "edit.clearFormats" | "home.clear")
            {
                let r_str = params.get("range").and_then(Value::as_str).unwrap_or("");
                return call_control(control_tx, ctx, "ai.deep_clear", json!({"range": r_str, "all": r_str.is_empty()}));
            }
            let res = exec(&resolved, params).map_err(|e| format!("command `{resolved}` failed: {e}"))?;
            if !turn_ctx.full_tool_access {
                let _ = call_control(control_tx, ctx, "ai.cleanup_orphaned_charts", json!({}));
            }
            Ok(json!({"command": resolved, "result": res}))
        }
        "execute_commands" => {
            let Some(items) = a.get("commands").and_then(Value::as_array) else {
                return Err("missing `commands`: pass an array of {command, params} entries".into());
            };
            if items.len() > 200 {
                return Err("too many commands in one batch (max 200); split the work into smaller batches".into());
            }
            let catalog = fetch_command_catalog(control_tx, ctx);
            let stop_on_error = a.get("stopOnError").and_then(Value::as_bool).unwrap_or(false);
            let mut results: Vec<Value> = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                let Some(obj) = item.as_object() else {
                    results.push(json!({"index": index, "ok": false, "error": "entry is not an object"}));
                    if stop_on_error {
                        break;
                    }
                    continue;
                };
                let raw = obj.get("command").or_else(|| obj.get("id")).and_then(Value::as_str).unwrap_or("");
                let resolved = if catalog.is_empty() {
                    raw.trim().to_string()
                } else {
                    match crate::ai_tools::resolve_command_id(&catalog, raw) {
                        Ok(id) => id,
                        Err(e) => {
                            results.push(json!({"index": index, "command": raw, "ok": false, "error": e}));
                            if stop_on_error {
                                break;
                            }
                            continue;
                        }
                    }
                };
                let params = crate::ai_tools::merge_command_params(obj);
                match exec(&resolved, params) {
                    Ok(v) => results.push(json!({"index": index, "command": resolved, "ok": true, "result": v})),
                    Err(e) => {
                        results.push(json!({"index": index, "command": resolved, "ok": false, "error": e}));
                        if stop_on_error {
                            break;
                        }
                    }
                }
            }
            let failed = results.iter().filter(|r| r.get("ok").and_then(Value::as_bool) != Some(true)).count();
            Ok(json!({"ran": results.len(), "failed": failed, "results": results}))
        }
        "list_functions" => {
            let mut p = json!({});
            if let Some(v) = a.get("search") {
                p["search"] = v.clone();
            }
            if let Some(v) = a.get("category") {
                p["category"] = v.clone();
            }
            exec("formulas.functions", p)
        }
        "select" => {
            if let Some(sheet) = a.get("sheet").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
                return exec("sheet.activate", json!({"sheet": sheet}));
            }
            let mut p = json!({});
            if let Some(range) = a.get("range").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
                p["range"] = json!(range);
            } else if let Some(cell) = a.get("cell").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
                p["cell"] = json!(cell);
            }
            exec("selection.set", p)
        }
        "sort_range" => {
            let mut p = json!({});
            if let Some(range) = a.get("range").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
                p["range"] = json!(range);
            }
            if let Some(keys) = a.get("keys") {
                p["keys"] = keys.clone();
                if let Some(header) = a.get("header") {
                    p["header"] = header.clone();
                }
                return exec("data.sort", p);
            }
            if let Some(column) = a.get("column") {
                p["column"] = column.clone();
            }
            let descending = a.get("order").and_then(Value::as_str).is_some_and(|o| o.eq_ignore_ascii_case("desc"));
            let command = if descending { "data.sortDescending" } else { "data.sortAscending" };
            exec(command, p)
        }
        "undo" => {
            let steps = a.get("steps").cloned().unwrap_or(json!(1));
            exec("edit.undo", json!({"steps": steps}))
        }
        "redo" => {
            let steps = a.get("steps").cloned().unwrap_or(json!(1));
            exec("edit.redo", json!({"steps": steps}))
        }
        "create_report" => create_report_tool(control_tx, ctx, a),
        "layout_sheet" => layout_sheet_tool(control_tx, ctx, a),
        "ui_control" => {
            if !turn_ctx.allow_ui_tools {
                return Err(
                    "`ui_control` is disabled in the AI Agent panel (turn on \"Tool UI/dialog\"). Use commands via execute_command instead.".into()
                );
            }
            let method = a.get("method").and_then(Value::as_str).ok_or("missing `method`")?;
            if !crate::ai_tools::ui_method_allowed(method) {
                return Err(format!(
                    "ui method `{method}` is not allowed from the agent. Allowed: {}. Prefer a GridCraft command via execute_command.",
                    crate::ai_tools::UI_METHODS.join(", ")
                ));
            }
            let params = a.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
            call_control(control_tx, ctx, method, params)
        }
        other => Err(format!(
            "unknown tool `{other}`. GridCraft features are commands: find one with list_commands (e.g. {{\"search\":\"sort\"}}) and run it with execute_command."
        )),
    }
}

// ============================================================================
// Layout engine: neutral, geometry-only final pass for `create_report` and `layout_sheet`.
// No fills, font colours, table styles or themes are applied here; only structure, widths,
// row heights, alignment, indent, number formats and functional horizontal borders.
// ============================================================================

/// Small command executor that records layout errors instead of failing the whole tool.
#[cfg(not(target_arch = "wasm32"))]
struct LayoutExec<'a> {
    control_tx: &'a Option<Sender<ControlRequest>>,
    ctx: &'a egui::Context,
    errors: Vec<String>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> LayoutExec<'a> {
    fn new(control_tx: &'a Option<Sender<ControlRequest>>, ctx: &'a egui::Context) -> Self {
        Self { control_tx, ctx, errors: Vec::new() }
    }

    fn run(&mut self, command: &str, params: Value) {
        if let Err(error) = call_control(self.control_tx, self.ctx, "engine.execute", json!({"command": command, "params": params})) {
            self.errors.push(format!("{command}: {error}"));
        }
    }

    fn read(&mut self, range: &str) -> Vec<Vec<Value>> {
        match call_control(self.control_tx, self.ctx, "sheet.read", json!({"range": range})) {
            Ok(reply) => reply
                .get("values")
                .and_then(Value::as_array)
                .map(|rows| rows.iter().map(|row| row.as_array().cloned().unwrap_or_default()).collect())
                .unwrap_or_default(),
            Err(error) => {
                self.errors.push(format!("sheet.read {range}: {error}"));
                Vec::new()
            }
        }
    }

    /// Insert blank rows so the following pass keeps every value/formula where it is semantically.
    fn insert_rows(&mut self, at_row: u32, count: u32) {
        self.run("home.insertRows", json!({"rows": format!("{at_row}:{at_row}"), "count": count}));
    }

    fn merge_across(&mut self, row: u32, last_col: &str) {
        self.run("home.mergeCells", json!({"range": format!("A{row}:{last_col}{row}")}));
    }

    fn row_height(&mut self, row: u32, points: f32) {
        self.run("home.rowHeight", json!({"rows": format!("{row}:{row}"), "points": points}));
    }

    fn border(&mut self, range: &str, preset: &str, style: &str) {
        self.run("home.borders", json!({"range": range, "preset": preset, "style": style}));
    }

    fn indent(&mut self, range: &str, indent: u8) {
        self.run("home.formatCells", json!({"range": range, "style": {"align": {"indent": indent}}}));
    }

    fn align(&mut self, range: &str, align: &str) {
        let command = match align {
            "Right" => "home.alignRight",
            "Center" => "home.alignCenter",
            _ => "home.alignLeft",
        };
        self.run(command, json!({"range": range}));
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct LayoutDataRow {
    /// Final 1-based sheet row.
    sheet_row: u32,
    role: crate::ai_tools::ReportRowRole,
    /// True when a section heading appeared above: line items are indented only inside a section.
    in_section: bool,
    values: Vec<Value>,
}

#[cfg(not(target_arch = "wasm32"))]
fn layout_number_columns(kinds: &[crate::ai_tools::ReportColumnKind]) -> Vec<usize> {
    kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| {
            matches!(
                kind,
                crate::ai_tools::ReportColumnKind::Money | crate::ai_tools::ReportColumnKind::Quantity | crate::ai_tools::ReportColumnKind::Percent
            )
        })
        .map(|(index, _)| index)
        .collect()
}

/// Apply the neutral table layout: functional borders, header styling, row hierarchy, alignment,
/// indent, number formats, widths, row heights, frozen header and print fit.
#[cfg(not(target_arch = "wasm32"))]
fn apply_table_layout(
    exec: &mut LayoutExec,
    headers: &[String],
    data_rows: &[LayoutDataRow],
    header_row: u32,
    ncols: usize,
) -> Vec<crate::ai_tools::ReportColumnKind> {
    let last_col = crate::ai_tools::column_letters(ncols as u32 - 1);
    let last_row = data_rows.iter().map(|row| row.sheet_row).max().unwrap_or(header_row);
    let data_start = header_row + 1;

    // Column kinds use the computed values (formulas already evaluated) and the header text.
    let kinds: Vec<crate::ai_tools::ReportColumnKind> = (0..ncols)
        .map(|col| {
            let values: Vec<Value> = data_rows.iter().filter_map(|row| row.values.get(col).cloned()).collect();
            crate::ai_tools::classify_report_column(headers.get(col).map(String::as_str).unwrap_or(""), &values)
        })
        .collect();

    // The whole block is vertically centred; horizontal alignment follows the column kind so the
    // header and every cell below it form one straight vertical line.
    exec.run("home.alignMiddle", json!({"range": format!("A{header_row}:{last_col}{last_row}")}));
    exec.border(&format!("A{header_row}:{last_col}{last_row}"), "none", "thin");
    exec.run("home.bold", json!({"range": format!("A{data_start}:{last_col}{last_row}"), "on": false}));
    exec.run("home.wrapText", json!({"range": format!("A{header_row}:{last_col}{header_row}"), "on": false}));

    // Header row: bold, top thin + bottom medium, height 25pt.
    let header_range = format!("A{header_row}:{last_col}{header_row}");
    exec.run("home.bold", json!({"range": header_range, "on": true}));
    exec.run("home.italic", json!({"range": header_range, "on": false}));
    exec.row_height(header_row, 25.0);
    exec.border(&header_range, "top", "thin");
    exec.border(&header_range, "bottom", "medium");
    for (col, kind) in kinds.iter().enumerate() {
        let letters = crate::ai_tools::column_letters(col as u32);
        exec.align(&format!("{letters}{header_row}"), kind.header_align());
        exec.align(&format!("{letters}{data_start}:{letters}{last_row}"), kind.data_align());
    }

    // Row hierarchy: section headings bold left, line items regular (indented inside a section),
    // subtotals bold with top+bottom thin borders, the grand total with a double bottom border.
    for row in data_rows {
        let full = format!("A{}:{last_col}{}", row.sheet_row, row.sheet_row);
        let label = format!("A{}:A{}", row.sheet_row, row.sheet_row);
        match row.role {
            crate::ai_tools::ReportRowRole::Section => {
                exec.indent(&label, 0);
                exec.run("home.bold", json!({"range": label, "on": true}));
                exec.row_height(row.sheet_row, 22.0);
            }
            crate::ai_tools::ReportRowRole::LineItem => {
                exec.indent(&label, if row.in_section { 1 } else { 0 });
                exec.row_height(row.sheet_row, 19.0);
            }
            crate::ai_tools::ReportRowRole::Subtotal => {
                exec.indent(&label, 0);
                exec.run("home.bold", json!({"range": full, "on": true}));
                exec.row_height(row.sheet_row, 22.0);
                exec.border(&full, "top", "thin");
                exec.border(&full, "bottom", "thin");
            }
            crate::ai_tools::ReportRowRole::GrandTotal => {
                exec.indent(&label, 0);
                exec.run("home.bold", json!({"range": full, "on": true}));
                exec.row_height(row.sheet_row, 24.0);
                exec.border(&full, "top", "thin");
                exec.border(&full, "bottom", "double");
            }
        }
    }

    // Number formats on the numeric columns (no `.00` for whole money/quantities, one decimal for
    // percentages), then widths from the FORMATTED display length with a safe +6 padding gap.
    for col in layout_number_columns(&kinds) {
        let letters = crate::ai_tools::column_letters(col as u32);
        let values: Vec<Value> = data_rows.iter().filter_map(|row| row.values.get(col).cloned()).collect();
        let format = kinds[col].number_format(crate::ai_tools::report_money_has_cents(&values));
        exec.run("home.numberFormat", json!({"range": format!("{letters}{data_start}:{letters}{last_row}"), "format": format}));
    }
    for (col, kind) in kinds.iter().enumerate() {
        let values: Vec<Value> = data_rows.iter().filter_map(|row| row.values.get(col).cloned()).collect();
        let width = crate::ai_tools::report_column_width(headers.get(col).map(String::as_str).unwrap_or(""), &values, *kind);
        let letters = crate::ai_tools::column_letters(col as u32);
        exec.run("home.columnWidth", json!({"cols": letters, "chars": width}));
    }

    // Freeze exactly below the column header row, and fit print output to one page wide.
    exec.run("view.freezePanes", json!({"cell": format!("A{}", header_row + 1)}));
    exec.run("pageLayout.printArea", json!({"range": format!("A1:{last_col}{last_row}")}));
    exec.run(
        "pageLayout.pageSetup",
        json!({
            "orientation": if ncols > 6 { "landscape" } else { "portrait" },
            "fitToPage": true,
            "fitWidth": 1,
            "fitHeight": 0
        }),
    );

    kinds
}

/// Style the merged title/subtitle block above the table. No border, no fill, left aligned (never
/// right), bold 15pt title and italic 11pt subtitle; exactly one 10pt blank spacer sits before the
/// header row (created by the caller through row insertion).
#[cfg(not(target_arch = "wasm32"))]
fn apply_title_block(exec: &mut LayoutExec, title_row: Option<u32>, subtitle_row: Option<u32>, spacer_row: Option<u32>, last_col: &str) {
    if let Some(row) = title_row {
        exec.merge_across(row, last_col);
        exec.align(&format!("A{row}"), "Left");
        exec.run("home.alignMiddle", json!({"range": format!("A{row}")}));
        exec.run("home.bold", json!({"range": format!("A{row}"), "on": true}));
        exec.run("home.italic", json!({"range": format!("A{row}"), "on": false}));
        exec.run("home.fontSize", json!({"range": format!("A{row}"), "size": 15}));
        exec.run("home.borders", json!({"range": format!("A{row}:{last_col}{row}"), "preset": "none"}));
        exec.row_height(row, 27.0);
    }
    if let Some(row) = subtitle_row {
        exec.merge_across(row, last_col);
        exec.align(&format!("A{row}"), "Left");
        exec.run("home.alignMiddle", json!({"range": format!("A{row}")}));
        exec.run("home.bold", json!({"range": format!("A{row}"), "on": false}));
        exec.run("home.italic", json!({"range": format!("A{row}"), "on": true}));
        exec.run("home.fontSize", json!({"range": format!("A{row}"), "size": 11}));
        exec.run("home.borders", json!({"range": format!("A{row}:{last_col}{row}"), "preset": "none"}));
        exec.row_height(row, 19.0);
    }
    if let Some(row) = spacer_row {
        exec.run("home.borders", json!({"range": format!("A{row}:{last_col}{row}"), "preset": "none"}));
        exec.row_height(row, 10.0);
    }
}

/// Build a clean report from scratch through the neutral layout engine.
#[cfg(not(target_arch = "wasm32"))]
fn create_report_tool(control_tx: &Option<Sender<ControlRequest>>, ctx: &egui::Context, a: &Map<String, Value>) -> Result<Value, String> {
    let headers: Vec<String> = a
        .get("headers")
        .and_then(Value::as_array)
        .ok_or("missing `headers`")?
        .iter()
        .map(|value| value.as_str().unwrap_or_default().to_string())
        .collect();
    if headers.is_empty() {
        return Err("`headers` is empty".into());
    }
    let rows_in: Vec<Vec<Value>> =
        a.get("rows").and_then(Value::as_array).ok_or("missing `rows`")?.iter().map(|row| row.as_array().cloned().unwrap_or_default()).collect();
    if rows_in.is_empty() {
        return Err("`rows` is empty".into());
    }
    let ncols = headers.len().max(rows_in.iter().map(Vec::len).max().unwrap_or(0)).max(2);
    let title = a.get("title").and_then(Value::as_str).unwrap_or("");
    let subtitle = a.get("subtitle").and_then(Value::as_str).unwrap_or("");
    let freeze = a.get("freeze_header").and_then(Value::as_bool).unwrap_or(true);
    let print = a.get("print_setup").and_then(Value::as_bool).unwrap_or(true);
    let last_col = crate::ai_tools::column_letters(ncols as u32 - 1);

    let mut exec = LayoutExec::new(control_tx, ctx);
    let padded_header: Vec<Value> = (0..ncols).map(|col| headers.get(col).map(|h| Value::String(h.clone())).unwrap_or(Value::Null)).collect();
    let padded_rows: Vec<Vec<Value>> =
        rows_in.iter().map(|data| (0..ncols).map(|col| data.get(col).cloned().unwrap_or(Value::Null)).collect()).collect();

    // 1) Values first: header + data rows are written contiguously from A1. Formulas are computed
    //    before the final layout pass so widths use the real display values.
    exec.run("range.setValues", json!({"range": "A1", "values": [padded_header]}));
    exec.run("range.setValues", json!({"range": "A2", "values": padded_rows}));

    // 2) Section spacer rows between a closed section (total row) and the next section heading.
    let labels: Vec<Option<String>> = padded_rows.iter().map(|row| row.first().and_then(Value::as_str).map(str::to_string)).collect();
    let sections = crate::ai_tools::report_section_flags(&padded_rows);
    let roles = crate::ai_tools::report_row_roles(&labels, &sections);
    let spacers = crate::ai_tools::report_spacer_after(&roles, &sections);
    for index in (0..spacers.len()).rev() {
        if spacers[index] {
            exec.insert_rows(2 + index as u32 + 1, 1);
        }
    }

    // 3) Title block above the table: title row, subtitle row and exactly one blank 10pt spacer
    //    before the header row. Inserting rows here keeps every formula reference correct.
    let has_title = !title.is_empty();
    let has_subtitle = !subtitle.is_empty();
    let top_rows = if has_title || has_subtitle { 1 + has_title as u32 + has_subtitle as u32 } else { 0 };
    if top_rows > 0 {
        exec.insert_rows(1, top_rows);
    }
    let header_row = 1 + top_rows;
    let data_start = header_row + 1;
    let title_row = has_title.then_some(1);
    let subtitle_row = has_subtitle.then_some(1 + has_title as u32);
    let spacer_row = (has_title || has_subtitle).then_some(1 + has_title as u32 + has_subtitle as u32);
    if has_title || has_subtitle {
        exec.run("range.setValues", json!({"range": "A1", "values": [[if has_title { title } else { subtitle }]]}));
        if has_title && has_subtitle {
            exec.run("range.setValues", json!({"range": "A2", "values": [[subtitle]]}));
        }
    }
    apply_title_block(&mut exec, title_row, subtitle_row, spacer_row, &last_col);

    // 4) Final positions (row mapping) and the computed values read back after all insertions.
    let mut final_rows = Vec::with_capacity(padded_rows.len());
    let mut inserted_before = 0u32;
    let mut last_row = data_start.saturating_sub(1);
    for (index, _) in padded_rows.iter().enumerate() {
        let sheet_row = data_start + index as u32 + inserted_before;
        final_rows.push(sheet_row);
        last_row = sheet_row;
        if spacers.get(index).copied().unwrap_or(false) {
            inserted_before += 1;
            exec.row_height(sheet_row + 1, 10.0);
        }
    }
    let read_values = exec.read(&format!("A{header_row}:{last_col}{last_row}"));
    let data_rows: Vec<LayoutDataRow> = padded_rows
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let offset = (final_rows[index] - header_row) as usize;
            LayoutDataRow {
                sheet_row: final_rows[index],
                role: roles.get(index).copied().unwrap_or(crate::ai_tools::ReportRowRole::LineItem),
                in_section: sections.iter().take(index).any(|flag| *flag),
                values: read_values.get(offset).cloned().unwrap_or_default(),
            }
        })
        .collect();

    let kinds = apply_table_layout(&mut exec, &headers, &data_rows, header_row, ncols);
    if !freeze {
        exec.run("view.freezePanes", json!({}));
    }
    if !print {
        exec.run("pageLayout.printArea", json!({"clear": true}));
    }

    // Optional neutral chart (default GridCraft palette; no report colours are forced).
    let mut chart_value = Value::Null;
    if let Some(chart) = a.get("chart").and_then(Value::as_object) {
        let chart_type = chart.get("type").and_then(Value::as_str).unwrap_or("column");
        let chart_title = chart.get("title").and_then(Value::as_str).unwrap_or(if title.is_empty() { "Chart" } else { title });
        let anchor = chart
            .get("at")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{}{}", crate::ai_tools::column_letters(ncols as u32 + 1), header_row));
        let mut ranges = vec![format!("A{data_start}:A{last_row}")];
        for col in layout_number_columns(&kinds).into_iter().take(3) {
            let letters = crate::ai_tools::column_letters(col as u32);
            ranges.push(format!("{letters}{data_start}:{letters}{last_row}"));
        }
        chart_value = json!({"range": ranges.join(","), "type": chart_type, "title": chart_title, "at": anchor});
        exec.run(
            "insert.chart",
            json!({"range": ranges.join(","), "type": chart_type, "title": chart_title, "at": anchor, "width": 520, "height": 320}),
        );
    }

    Ok(json!({
        "ok": exec.errors.is_empty(),
        "headerRow": header_row,
        "dataStartRow": data_start,
        "lastRow": last_row,
        "columns": ncols,
        "columnKinds": kinds.iter().map(|kind| format!("{kind:?}")).collect::<Vec<_>>(),
        "titleRows": title_row.into_iter().chain(subtitle_row).collect::<Vec<_>>(),
        "spacerRows": (0..spacers.len()).filter(|index| spacers[*index]).map(|index| final_rows[index] + 1).collect::<Vec<_>>(),
        "totalRows": data_rows.iter().filter(|row| matches!(row.role, crate::ai_tools::ReportRowRole::Subtotal | crate::ai_tools::ReportRowRole::GrandTotal)).map(|row| row.sheet_row).collect::<Vec<_>>(),
        "chart": chart_value,
        "errors": exec.errors
    }))
}

/// Repair an existing range with the same neutral layout pass, without changing any value/formula:
/// title merge/alignment, readable widths, row heights, indent hierarchy, number formats, functional
/// borders, frozen header and print fit. Missing blanks (below a subtitle, between sections) are
/// inserted with `home.insertRows`, which shifts formulas correctly.
#[cfg(not(target_arch = "wasm32"))]
fn layout_sheet_tool(control_tx: &Option<Sender<ControlRequest>>, ctx: &egui::Context, a: &Map<String, Value>) -> Result<Value, String> {
    let mut exec = LayoutExec::new(control_tx, ctx);
    let read_params = match a.get("range") {
        Some(range) => json!({"range": range}),
        None => json!({}),
    };
    let first = call_control(control_tx, ctx, "sheet.read", read_params)?;
    let range_text = first.get("range").and_then(Value::as_str).unwrap_or("A1").to_string();
    let parsed = RangeRef::parse(split_sheet(&range_text).1).ok_or_else(|| format!("cannot parse range `{range_text}`"))?;
    let start_col = parsed.start.col;
    let start_row = parsed.start.row + 1;
    let mut end_row = parsed.end.row + 1;
    let mut values: Vec<Vec<Value>> = first
        .get("values")
        .and_then(Value::as_array)
        .map(|rows| rows.iter().map(|row| row.as_array().cloned().unwrap_or_default()).collect())
        .unwrap_or_default();
    if values.is_empty() {
        return Err("the range is empty; write data first".into());
    }
    let ncols = values.iter().map(Vec::len).max().unwrap_or(1).max((parsed.end.col - parsed.start.col + 1) as usize).max(1);

    let non_empty =
        |row: &[Value]| row.iter().filter(|cell| !cell.is_null() && cell.as_str().map(|text| !text.trim().is_empty()).unwrap_or(true)).count();
    let mut header_row = a.get("header_row").and_then(Value::as_u64).map(|value| value as u32).unwrap_or_else(|| {
        values.iter().enumerate().find(|(_, row)| non_empty(row) >= 2).map(|(index, _)| start_row + index as u32).unwrap_or(start_row)
    });

    // One blank 10pt spacer below the subtitle, when the table header directly follows it.
    if header_row > start_row {
        let above_index = (header_row - start_row) as usize - 1;
        let above_empty = values.get(above_index).map(|row| non_empty(row) == 0).unwrap_or(true);
        if !above_empty && values.get((header_row - start_row) as usize).map(|row| non_empty(row)).unwrap_or(0) > 0 {
            exec.insert_rows(header_row, 1);
            header_row += 1;
            end_row += 1;
            let reread = exec.read(&format!("A{start_row}:{}{end_row}", crate::ai_tools::column_letters(start_col + ncols as u32 - 1)));
            if !reread.is_empty() {
                values = reread;
            }
        }
    }

    let data_start = header_row + 1;
    let read_rows: Vec<(u32, Vec<Value>)> = values
        .iter()
        .enumerate()
        .map(|(index, row)| (start_row + index as u32, row.clone()))
        .filter(|(sheet_row, _)| *sheet_row >= data_start && *sheet_row <= end_row)
        .collect();
    let row_labels: Vec<Option<String>> = read_rows.iter().map(|(_, row)| row.first().and_then(Value::as_str).map(str::to_string)).collect();
    let row_values: Vec<Vec<Value>> = read_rows.iter().map(|(_, row)| row.clone()).collect();
    let sections = crate::ai_tools::report_section_flags(&row_values);
    let roles = crate::ai_tools::report_row_roles(&row_labels, &sections);
    let spacers = crate::ai_tools::report_spacer_after(&roles, &sections);

    // Insert the section spacer rows from the bottom up so earlier positions stay valid.
    let mut inserted_after: Vec<u32> = vec![0; read_rows.len()];
    for index in (0..spacers.len()).rev() {
        if spacers[index] {
            exec.insert_rows(read_rows[index].0 + 1, 1);
            inserted_after[index] = 1;
            end_row += 1;
        }
    }
    let mut shift = 0u32;
    let mut spacer_rows: Vec<u32> = Vec::new();
    let mut data_rows = Vec::new();
    for (index, (sheet_row, row)) in read_rows.iter().enumerate() {
        let final_row = sheet_row + shift;
        if inserted_after[index] == 1 {
            exec.row_height(final_row + 1, 10.0);
            spacer_rows.push(final_row + 1);
            shift += 1;
        }
        data_rows.push(LayoutDataRow {
            sheet_row: final_row,
            role: roles.get(index).copied().unwrap_or(crate::ai_tools::ReportRowRole::LineItem),
            in_section: sections.iter().take(index).any(|flag| *flag),
            values: row.clone(),
        });
    }

    // Title/subtitle above the table: merge across the table, left aligned, no border, bold/italic.
    let title_rows: Vec<u32> = (start_row..header_row)
        .filter(|row| {
            let index = (row - start_row) as usize;
            values.get(index).is_some_and(|cells| non_empty(cells) > 0)
        })
        .collect();
    if let Some(title_row) = title_rows.first() {
        let last_col = crate::ai_tools::column_letters(start_col + ncols as u32 - 1);
        let subtitle_row = title_rows.get(1).copied();
        apply_title_block(&mut exec, Some(*title_row), subtitle_row, None, &last_col);
        if let Some(spacer) =
            (title_row + 1..header_row).find(|row| values.get((row - start_row) as usize).is_some_and(|cells| non_empty(cells) == 0))
        {
            exec.row_height(spacer, 10.0);
        }
    }

    let headers: Vec<String> = (0..ncols)
        .map(|col| values.get((header_row - start_row) as usize).and_then(|row| row.get(col)).and_then(Value::as_str).unwrap_or("").to_string())
        .collect();
    let kinds = apply_table_layout(&mut exec, &headers, &data_rows, header_row, ncols);

    Ok(json!({
        "ok": exec.errors.is_empty(),
        "range": format!("A{start_row}:{}{end_row}", crate::ai_tools::column_letters(start_col + ncols as u32 - 1)),
        "headerRow": header_row,
        "dataStartRow": header_row + 1,
        "lastRow": data_rows.iter().map(|row| row.sheet_row).max().unwrap_or(header_row),
        "columnKinds": kinds.iter().map(|kind| format!("{kind:?}")).collect::<Vec<_>>(),
        "totalRows": data_rows.iter().filter(|row| matches!(row.role, crate::ai_tools::ReportRowRole::Subtotal | crate::ai_tools::ReportRowRole::GrandTotal)).map(|row| row.sheet_row).collect::<Vec<_>>(),
        "spacerRows": spacer_rows,
        "errors": exec.errors
    }))
}

// ============================================================================
// Multi-Turn LLM Agent Loop (Cloud APIs & Local Ollama)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
fn run_api_agent_loop(
    mode: AgentMode,
    provider: ApiProvider,
    base_url: &str,
    api_key: &str,
    model: &str,
    history: Vec<(String, String)>,
    control_tx: Option<Sender<ControlRequest>>,
    tx: &Sender<WorkerEvent>,
    cancel: &AtomicBool,
    ctx: &egui::Context,
    is_local: bool,
    full_tool_access: bool,
    include_command_catalog: bool,
    allow_ui_tools: bool,
) {
    let last_user_prompt = history.iter().rev().find(|(role, _)| role == "user").map(|(_, content)| content.clone()).unwrap_or_default();
    let mut turn_ctx = TurnContext::new(&last_user_prompt, full_tool_access, allow_ui_tools);

    // Deterministic fast-path for explicit "hapus semua" / "clear all" requests so all data,
    // background colors, formal tables, auto-filters, selection boxes, and chart widgets are wiped immediately!
    // Full tool access deliberately skips it: the model decides which clear command (and target) the
    // request needs, instead of a keyword heuristic wiping the whole sheet.
    if !full_tool_access && prompt_wants_delete_all(&turn_ctx.user_prompt) && requested_chart_count(&turn_ctx.user_prompt).is_none() {
        let _ =
            tx.send(WorkerEvent::ToolStart { name: "clear_range".into(), args: "Clearing all data, fills, tables, filters and charts…".into() });
        let _ = call_control(&control_tx, ctx, "ai.deep_clear", json!({"all": true}));
        let _ = tx.send(WorkerEvent::ToolDone {
            name: "clear_range".into(),
            summary: "Cleared data, fills, tables, filters and chart widgets".into(),
            is_error: false,
        });
        let _ = tx.send(WorkerEvent::AssistantReply("Cleared all data, fills, tables, filters and chart widgets on the active sheet.".into()));
        ctx.request_repaint();
        return;
    }

    // Capture Table 1's style & formula blueprint before the turn executes (legacy merge helper).
    if !full_tool_access {
        let _ = call_control(&control_tx, ctx, "ai.capture_blueprint", json!({}));
    }

    let live_wb_context = build_live_workbook_context(&control_tx, ctx);
    let local_note = if is_local {
        // Local models have less context and weaker tool calling: tell them explicitly that the
        // context below is complete and that they must act through tools, one step at a time.
        "LOCAL MODEL NOTE: the live workbook context and command catalog below are complete for the active sheet. Read them before acting, use tools for every change, and take one step at a time.\n\n"
    } else {
        ""
    };
    let access_section = if full_tool_access {
        let catalog_text = if include_command_catalog {
            let catalog = fetch_command_catalog(&control_tx, ctx);
            // Light prompts: local models get only the compact catalog (the ground rules carry the
            // task protocol); cloud/terminal agents get the concise guide + recipes + catalog.
            crate::ai_tools::catalog_prompt(&catalog, if is_local { 5000 } else { 12000 })
        } else {
            String::new()
        };
        if is_local { catalog_text } else { format!("{}\n\n{}\n\n{catalog_text}", crate::ai_tools::FEATURE_GUIDE, crate::ai_tools::CHEAT_SHEET) }
    } else {
        String::new()
    };
    let full_system_prompt = format!("{}\n\n{SYSTEM_PROMPT}\n\n{local_note}{access_section}\n\n{live_wb_context}", crate::ai_tools::GROUND_RULES);

    let mut messages = vec![json!({"role": "system", "content": full_system_prompt})];
    for (role, content) in history {
        messages.push(json!({"role": role, "content": content}));
    }

    let tools = crate::ai_tools::openai_tools(full_tool_access, allow_ui_tools);
    let max_steps = if full_tool_access { 24 } else { 12 };
    let mut replied = false;
    let endpoint = if provider == ApiProvider::Anthropic { format!("{base_url}/messages") } else { format!("{base_url}/chat/completions") };
    let mut use_native_tools = true;

    for _step in 0..max_steps {
        if cancel.load(Ordering::SeqCst) {
            return;
        }
        if is_local {
            if !check_endpoint_reachable(base_url) {
                let _ = tx.send(WorkerEvent::ConnectionLost { mode, reason: "Local Ollama server stopped / unreachable".into() });
                return;
            }
        } else if !check_internet_online(Some(base_url)) {
            let _ = tx.send(WorkerEvent::ConnectionLost { mode, reason: "No internet connection".into() });
            return;
        }

        let body = if provider == ApiProvider::Anthropic {
            let anth_tools: Vec<Value> = tools
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|t| {
                    json!({
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"]
                    })
                })
                .collect();
            let anth_msgs: Vec<Value> = messages.iter().skip(1).cloned().collect();
            if use_native_tools {
                json!({
                    "model": model,
                    "max_tokens": 4096,
                    "system": full_system_prompt,
                    "messages": anth_msgs,
                    "tools": anth_tools
                })
            } else {
                json!({
                    "model": model,
                    "max_tokens": 4096,
                    "system": full_system_prompt,
                    "messages": anth_msgs
                })
            }
        } else if use_native_tools {
            json!({
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            })
        } else {
            json!({
                "model": model,
                "messages": messages
            })
        };

        let resp = match http_post_json(&endpoint, provider, api_key, &body, cancel, is_local, 0) {
            Ok(v) => v,
            Err(e) => {
                let lower = e.to_ascii_lowercase();
                if use_native_tools && (lower.contains("tool") || lower.contains("function") || lower.contains("schema")) {
                    // The provider rejected the request because of the tool schema: retry once
                    // without native tools and let the `<gridcraft_tool>` text protocol carry the
                    // calls (common with older Ollama builds and some free API models).
                    use_native_tools = false;
                    continue;
                }
                if !is_local && !check_internet_online(Some(base_url)) {
                    let _ = tx.send(WorkerEvent::ConnectionLost { mode, reason: "No internet connection".into() });
                } else {
                    let _ = tx.send(WorkerEvent::Error(e));
                }
                return;
            }
        };

        if cancel.load(Ordering::SeqCst) {
            return;
        }

        if let Some(err) = resp.get("error") {
            let msg = err.get("message").and_then(Value::as_str).or_else(|| err.as_str()).unwrap_or("API error");
            // If the selected model doesn't support native function calling (common on some local
            // Ollama or free OpenRouter models), retry immediately without `tools` and use the
            // `<gridcraft_tool>` tag extraction instead.
            let lower = msg.to_ascii_lowercase();
            if use_native_tools && (lower.contains("tool") || lower.contains("function") || lower.contains("schema") || lower.contains("unsupported"))
            {
                use_native_tools = false;
                continue;
            }
            let _ = tx.send(WorkerEvent::Error(msg.to_string()));
            return;
        }

        // Parse OpenAI-compatible response (also used by Ollama /v1/chat/completions)
        if let Some(choice) = resp.get("choices").and_then(Value::as_array).and_then(|c| c.first()) {
            let msg = &choice["message"];
            let content_text = msg.get("content").and_then(Value::as_str).unwrap_or("").trim().to_string();
            let tool_calls = msg.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();

            if tool_calls.is_empty() {
                let clean = if content_text.contains("<gridcraft_tool>") {
                    extract_and_run_gridcraft_tools(&content_text, &control_tx, tx, ctx, cancel, &mut turn_ctx)
                } else {
                    content_text
                };
                if !turn_ctx.full_tool_access {
                    ensure_post_turn_consistency(&control_tx, tx, ctx, cancel, &mut turn_ctx);
                }
                if !turn_ctx.tool_errors.is_empty() && turn_ctx.tool_success_count == 0 {
                    let _ = tx.send(WorkerEvent::Error(turn_ctx.tool_errors.join("; ")));
                    return;
                }
                if !clean.is_empty() {
                    let _ = tx.send(WorkerEvent::AssistantReply(clean));
                } else {
                    let _ = tx.send(WorkerEvent::AssistantReply("Done.".into()));
                }
                return;
            }

            if !content_text.is_empty() {
                let clean = extract_and_run_gridcraft_tools(&content_text, &control_tx, tx, ctx, cancel, &mut turn_ctx);
                if !clean.is_empty() {
                    replied = true;
                    let _ = tx.send(WorkerEvent::AssistantReply(clean));
                    ctx.request_repaint();
                }
            }

            messages.push(msg.clone());

            for tc in tool_calls {
                if cancel.load(Ordering::SeqCst) {
                    return;
                }
                let call_id = tc.get("id").and_then(Value::as_str).unwrap_or("call_1").to_string();
                let fn_name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let raw_args = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let parsed_args = match &tc["function"]["arguments"] {
                    Value::Object(_) => Some(tc["function"]["arguments"].clone()),
                    _ => serde_json::from_str::<Value>(raw_args).ok().filter(Value::is_object),
                };
                let Some(parsed_args) = parsed_args else {
                    // Never run a command with guessed/empty params: that can clear or reformat the
                    // wrong cells. Report the malformed arguments so the model can correct itself.
                    let err = format!(
                        "tool `{fn_name}` was not run: `arguments` was not a JSON object ({}).",
                        raw_args.chars().take(160).collect::<String>()
                    );
                    turn_ctx.tool_errors.push(err.clone());
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json!({"error": err}).to_string()
                    }));
                    let _ = tx.send(WorkerEvent::ToolDone { name: fn_name, summary: format!("Failed: {err}"), is_error: true });
                    ctx.request_repaint();
                    continue;
                };

                let _ = tx.send(WorkerEvent::ToolStart { name: fn_name.clone(), args: humanize_tool_start(&fn_name, &parsed_args) });
                ctx.request_repaint();

                let result = execute_mcp_tool(&control_tx, ctx, &fn_name, &parsed_args, &mut turn_ctx);
                let (tool_output, is_err, summary) = match result {
                    Ok(v) => {
                        turn_ctx.tool_success_count += 1;
                        let s = crate::ai_tools::truncate_tool_output(&serde_json::to_string(&v).unwrap_or_default(), 12000);
                        let human = humanize_tool_done(&fn_name, &parsed_args, &v);
                        (s, false, human)
                    }
                    Err(e) => {
                        turn_ctx.tool_errors.push(e.clone());
                        (json!({"error": e}).to_string(), true, format!("Failed: {e}"))
                    }
                };

                let _ = tx.send(WorkerEvent::ToolDone { name: fn_name, summary, is_error: is_err });
                ctx.request_repaint();

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_output
                }));
            }
            continue;
        }

        // Parse Anthropic Messages response
        if let Some(blocks) = resp.get("content").and_then(Value::as_array) {
            let mut text_parts = Vec::new();
            let mut tool_uses = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            text_parts.push(t.to_string());
                        }
                    }
                    Some("tool_use") => tool_uses.push(b.clone()),
                    _ => {}
                }
            }
            let combined = text_parts.join("\n").trim().to_string();
            if tool_uses.is_empty() {
                let clean = extract_and_run_gridcraft_tools(&combined, &control_tx, tx, ctx, cancel, &mut turn_ctx);
                if !turn_ctx.full_tool_access {
                    ensure_post_turn_consistency(&control_tx, tx, ctx, cancel, &mut turn_ctx);
                }
                if !turn_ctx.tool_errors.is_empty() && turn_ctx.tool_success_count == 0 {
                    let _ = tx.send(WorkerEvent::Error(turn_ctx.tool_errors.join("; ")));
                    return;
                }
                let _ = tx.send(WorkerEvent::AssistantReply(if clean.is_empty() { "Done.".into() } else { clean }));
                return;
            }
            if !combined.is_empty() {
                replied = true;
                let _ = tx.send(WorkerEvent::AssistantReply(combined));
            }
            messages.push(json!({"role": "assistant", "content": blocks}));
            let mut tool_results = Vec::new();
            for tu in tool_uses {
                if cancel.load(Ordering::SeqCst) {
                    return;
                }
                let id = tu.get("id").and_then(Value::as_str).unwrap_or("tu_1").to_string();
                let fn_name = tu.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let input = tu.get("input").cloned().unwrap_or_else(|| json!({}));
                let _ = tx.send(WorkerEvent::ToolStart { name: fn_name.clone(), args: humanize_tool_start(&fn_name, &input) });
                ctx.request_repaint();
                let res = execute_mcp_tool(&control_tx, ctx, &fn_name, &input, &mut turn_ctx);
                let (out_str, is_err, summary) = match res {
                    Ok(v) => {
                        turn_ctx.tool_success_count += 1;
                        let s = crate::ai_tools::truncate_tool_output(&serde_json::to_string(&v).unwrap_or_default(), 12000);
                        let human = humanize_tool_done(&fn_name, &input, &v);
                        (s, false, human)
                    }
                    Err(e) => {
                        turn_ctx.tool_errors.push(e.clone());
                        (json!({"error": e}).to_string(), true, format!("Failed: {e}"))
                    }
                };
                let _ = tx.send(WorkerEvent::ToolDone { name: fn_name, summary, is_error: is_err });
                ctx.request_repaint();
                tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": out_str
                }));
            }
            messages.push(json!({"role": "user", "content": tool_results}));
            continue;
        }

        let _ = tx.send(WorkerEvent::Error("Unrecognized API response format.".into()));
        return;
    }

    // The step budget ran out while the model kept calling tools: close the turn with a message
    // instead of leaving the chat without an answer (the executed commands stay applied).
    if !replied {
        let _ = tx.send(WorkerEvent::AssistantReply(format!(
            "Done — reached the {max_steps} step limit. Commands that succeeded were applied; ask me to continue if more is needed."
        )));
        ctx.request_repaint();
    }
}

// ============================================================================
// Terminal CLI Agent Runner (`agy`, `claude`, `codex`)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
fn cli_agent_args(cli_agent: CliAgent, cli_model: &str, prompt: &str) -> (Vec<String>, bool) {
    match cli_agent {
        CliAgent::Agy => {
            ensure_clean_agy_mcp_config();
            let mut a = vec![
                "-p".into(),
                prompt.to_string(),
                "--disable-slash-commands".into(),
                "--dangerously-skip-permissions".into(),
                "--print-timeout".into(),
                "3600s".into(),
            ];
            if !cli_model.is_empty() && cli_model != "default" {
                a.push("--model".into());
                a.push(cli_model.to_string());
            }
            (a, false)
        }
        CliAgent::Claude => {
            let mut a = vec!["-p".into(), "--tools".into(), "".into(), "--strict-mcp-config".into(), "--no-session-persistence".into()];
            if !cli_model.is_empty() && cli_model != "default" {
                a.push("--model".into());
                a.push(cli_model.to_string());
            }
            (a, true)
        }
        CliAgent::Codex => {
            let mut a = vec![
                "exec".into(),
                "-c".into(),
                "mcp_servers={}".into(),
                "-c".into(),
                "model_reasoning_effort=\"medium\"".into(),
                "--skip-git-repo-check".into(),
            ];
            if !cli_model.is_empty() && cli_model != "default" {
                a.push("-m".into());
                a.push(cli_model.to_string());
            }
            a.push("-".into());
            (a, true)
        }
        CliAgent::Custom => (vec!["-p".into(), prompt.to_string()], false),
    }
}

/// Build the Terminal agent prompt. With full access the prompt carries the same command catalog,
/// recipes and tool list as the cloud/API path, because a Terminal CLI agent cannot call back into
/// GridCraft mid-turn: everything it needs to succeed has to arrive in this one prompt.
#[cfg(not(target_arch = "wasm32"))]
fn build_terminal_prompt(
    live_wb_context: &str,
    port_note: &str,
    user_prompt: &str,
    full_tool_access: bool,
    include_command_catalog: bool,
    allow_ui_tools: bool,
    control_tx: &Option<Sender<ControlRequest>>,
    ctx: &egui::Context,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(crate::ai_tools::GROUND_RULES);
    prompt.push('\n');
    prompt.push_str("You are the built-in AI Agent controlling a live GridCraft spreadsheet.\n");
    prompt.push_str(
        "IMPORTANT: Do NOT run any shell, bash, file, or directory commands! Do NOT inspect the filesystem. Respond immediately in text!\n",
    );
    prompt.push_str(
        "RESPONSE STYLE: Keep your text reply VERY SHORT, CONCISE, and TIDY (1 to 3 short sentences or bullet lines using `•`). NEVER use asterisk (`*` or `**`) characters in your reply!\n\n",
    );
    if full_tool_access {
        prompt.push_str("You have FULL access to every GridCraft feature through tool calls.\n\n");
        prompt.push_str(crate::ai_tools::FEATURE_GUIDE);
        prompt.push_str("\n\n");
        prompt.push_str(crate::ai_tools::CHEAT_SHEET);
        prompt.push_str("\n\n");
        if include_command_catalog {
            let catalog = fetch_command_catalog(control_tx, ctx);
            prompt.push_str(&crate::ai_tools::catalog_prompt(&catalog, 12000));
            prompt.push('\n');
        }
    }
    prompt.push_str(live_wb_context);
    prompt.push('\n');
    if !port_note.is_empty() {
        prompt.push_str(port_note);
        prompt.push('\n');
    }
    let tools = crate::ai_tools::tool_names(full_tool_access, allow_ui_tools);
    prompt.push_str(&format!(
        "Available tools (emit any of them as <gridcraft_tool>{{\"name\":\"<tool>\",\"arguments\":{{...}}}}</gridcraft_tool>): {}\n",
        tools.join(", ")
    ));
    prompt.push_str(
        "Examples (valid JSON inside the tags):\n\
- Write table data: <gridcraft_tool>{\"name\":\"write_range\",\"arguments\":{\"range\":\"A1\",\"values\":[[\"Produk\",\"Q1\",\"Q2\"],[\"Laptop\",120,150]]}}</gridcraft_tool>\n",
    );
    prompt.push_str(
        "- Run ANY GridCraft feature (sort, filter, freeze, pivot, chart, conditional formatting, page setup, print, validation, protection, comment, file save/export, ...): <gridcraft_tool>{\"name\":\"execute_command\",\"arguments\":{\"command\":\"data.sortDescending\",\"params\":{\"range\":\"A1:C10\",\"column\":\"B\"}}}</gridcraft_tool>\n",
    );
    prompt.push_str(
        "- Run several commands: <gridcraft_tool>{\"name\":\"execute_commands\",\"arguments\":{\"commands\":[{\"command\":\"home.bold\",\"params\":{\"range\":\"A1:F1\"}},{\"command\":\"home.fillColor\",\"params\":{\"range\":\"A1:F1\",\"color\":\"#1F9D55\"}}]}}</gridcraft_tool>\n",
    );
    prompt.push_str(
        "- Discover ids/params: <gridcraft_tool>{\"name\":\"list_commands\",\"arguments\":{\"search\":\"filter\"}}</gridcraft_tool> and <gridcraft_tool>{\"name\":\"describe_command\",\"arguments\":{\"command\":\"data.filter\"}}</gridcraft_tool>\n",
    );
    prompt.push_str(
        "- Insert a chart: <gridcraft_tool>{\"name\":\"insert_chart\",\"arguments\":{\"range\":\"B4:B10,C4:C10\",\"type\":\"column\",\"title\":\"Q1 vs Q2\",\"at\":\"K2\"}}</gridcraft_tool>\n",
    );
    prompt.push_str("- Clear data: <gridcraft_tool>{\"name\":\"clear_range\",\"arguments\":{\"all\":true}}</gridcraft_tool>\n\n");
    prompt.push_str("User request: ");
    prompt.push_str(user_prompt);
    prompt
}

#[cfg(not(target_arch = "wasm32"))]
fn run_cli_agent_turn(
    cli_agent: CliAgent,
    cmd_name: &str,
    cli_model: &str,
    user_prompt: &str,
    control_tx: Option<Sender<ControlRequest>>,
    control_port: Option<u16>,
    tx: &Sender<WorkerEvent>,
    cancel: &AtomicBool,
    ctx: &egui::Context,
    full_tool_access: bool,
    include_command_catalog: bool,
    allow_ui_tools: bool,
) {
    let mut turn_ctx = TurnContext::new(user_prompt, full_tool_access, allow_ui_tools);

    // Deterministic fast-path for explicit "hapus semua" / "clear all" requests. Full tool access
    // skips it so the model decides which clear command (and target) the request needs.
    if !full_tool_access && prompt_wants_delete_all(&turn_ctx.user_prompt) && requested_chart_count(&turn_ctx.user_prompt).is_none() {
        let _ =
            tx.send(WorkerEvent::ToolStart { name: "clear_range".into(), args: "Clearing all data, fills, tables, filters and charts…".into() });
        let _ = call_control(&control_tx, ctx, "ai.deep_clear", json!({"all": true}));
        let _ = tx.send(WorkerEvent::ToolDone {
            name: "clear_range".into(),
            summary: "Cleared data, fills, tables, filters and chart widgets".into(),
            is_error: false,
        });
        let _ = tx.send(WorkerEvent::AssistantReply("Cleared all data, fills, tables, filters and chart widgets on the active sheet.".into()));
        ctx.request_repaint();
        return;
    }

    if !full_tool_access {
        let _ = call_control(&control_tx, ctx, "ai.capture_blueprint", json!({}));
    }
    let live_wb_context = build_live_workbook_context(&control_tx, ctx);
    let port_note = match control_port {
        Some(p) => format!("A live GridCraft control port is also listening on 127.0.0.1:{p}."),
        None => String::new(),
    };
    let base_prompt =
        build_terminal_prompt(&live_wb_context, &port_note, user_prompt, full_tool_access, include_command_catalog, allow_ui_tools, &control_tx, ctx);

    let work_dir = isolated_agent_workspace();
    let mut prompt_for_attempt = base_prompt.clone();
    // Two attempts at most: a failed first attempt (every tool errored) gets one chance to fix its
    // arguments with our errors in the prompt. Successful commands are never replayed.
    for attempt in 1..=2_u32 {
        let (args, use_stdin) = cli_agent_args(cli_agent, cli_model, &prompt_for_attempt);
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let stdin_bytes = if use_stdin { Some(prompt_for_attempt.as_bytes()) } else { None };
        match run_process_cancellable_with_stdin(cmd_name, &arg_refs, stdin_bytes, work_dir.as_deref(), cancel, false, 0) {
            Ok(output) => {
                if cancel.load(Ordering::SeqCst) {
                    return;
                }
                if !output.contains("<gridcraft_tool>")
                    && let Some(err_msg) = detect_cli_output_error(&output)
                {
                    let _ = tx.send(WorkerEvent::Error(err_msg));
                    return;
                }
                let clean_reply = extract_and_run_gridcraft_tools(&output, &control_tx, tx, ctx, cancel, &mut turn_ctx);
                if !turn_ctx.full_tool_access {
                    ensure_post_turn_consistency(&control_tx, tx, ctx, cancel, &mut turn_ctx);
                }
                if !turn_ctx.tool_errors.is_empty() && turn_ctx.tool_success_count == 0 {
                    if attempt == 1 {
                        // Give the agent one round to correct the command ids/arguments it got wrong.
                        let errors = turn_ctx.tool_errors.join("\n");
                        turn_ctx.tool_errors.clear();
                        turn_ctx.tool_success_count = 0;
                        prompt_for_attempt.push_str(
                            "\n\n=== PREVIOUS ATTEMPT FAILED: every tool call errored. Fix the command ids/params and emit corrected <gridcraft_tool> tags. Errors were:\n",
                        );
                        prompt_for_attempt.push_str(&errors);
                        prompt_for_attempt.push('\n');
                        continue;
                    }
                    let _ = tx.send(WorkerEvent::Error(turn_ctx.tool_errors.join("; ")));
                    return;
                }
                let final_text = clean_reply.trim();
                if !final_text.is_empty() {
                    let _ = tx.send(WorkerEvent::AssistantReply(final_text.to_string()));
                } else {
                    let _ = tx.send(WorkerEvent::AssistantReply("Done.".into()));
                }
                return;
            }
            Err(e) => {
                if !check_internet_online(None) {
                    let _ = tx.send(WorkerEvent::ConnectionLost { mode: AgentMode::Terminal, reason: "No internet connection detected".into() });
                } else if !cancel.load(Ordering::SeqCst) {
                    let _ = tx.send(WorkerEvent::Error(e));
                }
                return;
            }
        }
    }
}

/// Extract and execute all `<gridcraft_tool>...</gridcraft_tool>` blocks from an assistant response,
/// returning the remaining human-readable text.
#[cfg(not(target_arch = "wasm32"))]
fn extract_and_run_gridcraft_tools(
    output: &str,
    control_tx: &Option<Sender<ControlRequest>>,
    tx: &Sender<WorkerEvent>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
    turn_ctx: &mut TurnContext,
) -> String {
    let mut clean_reply = String::new();
    let mut rest = output;
    while let Some(start) = rest.find("<gridcraft_tool>") {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        clean_reply.push_str(&rest[..start]);
        let after_start = &rest[start + "<gridcraft_tool>".len()..];
        if let Some(end) = after_start.find("</gridcraft_tool>") {
            let json_str = after_start[..end].trim();
            if let Some(val) = parse_tool_json_flexible(json_str) {
                let name = val.get("name").and_then(Value::as_str).unwrap_or("execute_command").to_string();
                let t_args = val.get("arguments").cloned().unwrap_or_else(|| json!({}));
                let _ = tx.send(WorkerEvent::ToolStart { name: name.clone(), args: humanize_tool_start(&name, &t_args) });
                ctx.request_repaint();
                let res = execute_mcp_tool(control_tx, ctx, &name, &t_args, turn_ctx);
                let (summary, is_error) = match res {
                    Ok(v) => {
                        turn_ctx.tool_success_count += 1;
                        (humanize_tool_done(&name, &t_args, &v), false)
                    }
                    Err(e) => {
                        turn_ctx.tool_errors.push(e.clone());
                        (format!("Failed: {e}"), true)
                    }
                };
                let _ = tx.send(WorkerEvent::ToolDone { name, summary, is_error });
                ctx.request_repaint();
            }
            rest = &after_start[end + "</gridcraft_tool>".len()..];
        } else {
            clean_reply.push_str(after_start);
            rest = "";
        }
    }
    clean_reply.push_str(rest);
    clean_reply.trim().to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_tool_json_flexible(raw: &str) -> Option<Value> {
    let trimmed = raw.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    None
}

// ============================================================================
// Subprocess & HTTP Helpers (Non-Blocking Threaded Readers + Process Tree Kill)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
// Native-only code: the workspace forbids clock reads on wasm because they trap there.
#[allow(clippy::disallowed_methods)]
fn http_post_json(
    url: &str,
    provider: ApiProvider,
    api_key: &str,
    body: &Value,
    cancel: &AtomicBool,
    is_local: bool,
    timeout_secs: u64,
) -> Result<Value, String> {
    let temp_dir = std::env::temp_dir();
    let id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let req_path = temp_dir.join(format!("gc_ai_req_{id}.json"));
    let body_bytes = serde_json::to_vec(body).map_err(|e| e.to_string())?;
    std::fs::write(&req_path, &body_bytes).map_err(|e| format!("temp file write: {e}"))?;

    // A timeout of 0 means "no request timeout": a slow local model may take as long as it needs.
    // Cancel/Stop still aborts the process, and real HTTP/network/model errors are still reported.
    let mut args: Vec<String> = vec!["-sS".into()];
    if timeout_secs > 0 {
        args.push("--max-time".into());
        args.push(timeout_secs.to_string());
    }
    args.extend(["-X".into(), "POST".into(), url.to_string(), "-H".into(), "Content-Type: application/json".into()]);

    if provider == ApiProvider::Anthropic {
        args.push("-H".into());
        args.push(format!("x-api-key: {api_key}"));
        args.push("-H".into());
        args.push("anthropic-version: 2023-06-01".into());
    } else if !api_key.is_empty() {
        args.push("-H".into());
        args.push(format!("Authorization: Bearer {api_key}"));
    }

    if provider == ApiProvider::OpenRouter {
        args.push("-H".into());
        args.push("HTTP-Referer: https://github.com/sergioregar/Morbid".into());
        args.push("-H".into());
        args.push("X-Title: GridCraft".into());
    }

    args.push("--data-binary".into());
    args.push(format!("@{}", req_path.display()));

    let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let res = run_process_cancellable_with_stdin(
        curl_bin,
        &arg_refs,
        None,
        None,
        cancel,
        is_local,
        if timeout_secs == 0 { 0 } else { timeout_secs.saturating_add(5) },
    );
    let _ = std::fs::remove_file(&req_path);

    let out = res?;
    serde_json::from_str::<Value>(&out).map_err(|e| {
        let preview: String = out.chars().take(180).collect();
        format!("Invalid JSON response ({e}): {preview}")
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn configure_no_window(cmd: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

#[cfg(not(target_arch = "wasm32"))]
fn kill_process_tree(child: &mut std::process::Child) {
    #[cfg(windows)]
    {
        let pid = child.id().to_string();
        let mut tk = std::process::Command::new("taskkill");
        tk.args(["/F", "/T", "/PID", &pid])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        configure_no_window(&mut tk);
        let _ = tk.status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_cli_command(program: &str, args: &[&str], has_stdin: bool, work_dir: Option<&std::path::Path>) -> std::io::Result<std::process::Child> {
    use std::process::{Command, Stdio};

    let make_stdin = || if has_stdin { Stdio::piped() } else { Stdio::null() };

    let mut direct = Command::new(program);
    direct.args(args).stdin(make_stdin()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = work_dir {
        direct.current_dir(dir);
    }
    configure_no_window(&mut direct);
    match direct.spawn() {
        Ok(child) => Ok(child),
        #[cfg(windows)]
        Err(_) => {
            // Try `<program>.cmd` first on Windows (for npm global scripts like `claude.cmd` and `codex.cmd`)
            let cmd_shim = format!("{program}.cmd");
            let mut shim = Command::new("cmd");
            shim.arg("/c").arg(&cmd_shim).args(args).stdin(make_stdin()).stdout(Stdio::piped()).stderr(Stdio::piped());
            if let Some(dir) = work_dir {
                shim.current_dir(dir);
            }
            configure_no_window(&mut shim);
            shim.spawn()
        }
        #[cfg(not(windows))]
        Err(e) => Err(e),
    }
}

/// Spawn a subprocess, optionally write `stdin_data` to its stdin, read `stdout` and `stderr`
/// concurrently on background threads so OS pipe buffers never deadlock and grandchild processes
/// inheriting pipes never hang `wait_with_output()`.
#[cfg(not(target_arch = "wasm32"))]
// Native-only code: the workspace forbids clock reads on wasm because they trap there.
#[allow(clippy::disallowed_methods)]
fn run_process_cancellable_with_stdin(
    program: &str,
    args: &[&str],
    stdin_data: Option<&[u8]>,
    work_dir: Option<&std::path::Path>,
    cancel: &AtomicBool,
    is_local: bool,
    timeout_secs: u64,
) -> Result<String, String> {
    use std::io::{Read, Write};
    use std::time::{Duration, Instant};

    let mut child = spawn_cli_command(program, args, stdin_data.is_some(), work_dir).map_err(|e| format!("Failed to launch `{program}`: {e}"))?;

    if let Some(bytes) = stdin_data
        && let Some(mut sin) = child.stdin.take()
    {
        let data = bytes.to_vec();
        let _ = std::thread::spawn(move || {
            let _ = sin.write_all(&data);
            let _ = sin.flush();
        });
    }

    let stdout_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let stdout_done = Arc::new(AtomicBool::new(false));
    let stderr_done = Arc::new(AtomicBool::new(false));

    if let Some(mut out_pipe) = child.stdout.take() {
        let buf = Arc::clone(&stdout_buf);
        let done = Arc::clone(&stdout_done);
        let _ = std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            while let Ok(n) = out_pipe.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if let Ok(mut guard) = buf.lock() {
                    guard.extend_from_slice(&chunk[..n]);
                }
            }
            done.store(true, Ordering::SeqCst);
        });
    } else {
        stdout_done.store(true, Ordering::SeqCst);
    }

    if let Some(mut err_pipe) = child.stderr.take() {
        let buf = Arc::clone(&stderr_buf);
        let done = Arc::clone(&stderr_done);
        let _ = std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            while let Ok(n) = err_pipe.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if let Ok(mut guard) = buf.lock() {
                    guard.extend_from_slice(&chunk[..n]);
                }
            }
            done.store(true, Ordering::SeqCst);
        });
    } else {
        stderr_done.store(true, Ordering::SeqCst);
    }

    let start = Instant::now();
    let mut last_net_check = Instant::now();
    let max_dur = (timeout_secs > 0).then(|| Duration::from_secs(timeout_secs.max(5)));

    loop {
        if cancel.load(Ordering::SeqCst) {
            kill_process_tree(&mut child);
            return Err("Task was cancelled.".into());
        }
        if !is_local && last_net_check.elapsed() >= Duration::from_secs(3) {
            last_net_check = Instant::now();
            if !check_internet_online(None) {
                kill_process_tree(&mut child);
                return Err("Internet connection lost; active task stopped.".into());
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                // Wait up to 250ms for reader threads to drain buffered output, without blocking if a daemon kept the pipe open
                let drain_start = Instant::now();
                while (!stdout_done.load(Ordering::SeqCst) || !stderr_done.load(Ordering::SeqCst))
                    && drain_start.elapsed() < Duration::from_millis(250)
                {
                    std::thread::sleep(Duration::from_millis(15));
                }
                let stdout = stdout_buf.lock().map(|g| String::from_utf8_lossy(&g).trim().to_string()).unwrap_or_default();
                let stderr = stderr_buf.lock().map(|g| String::from_utf8_lossy(&g).trim().to_string()).unwrap_or_default();

                if status.success() || !stdout.is_empty() {
                    return Ok(if stdout.is_empty() { stderr } else { stdout });
                }
                return Err(if stderr.is_empty() { format!("Process `{program}` exited with status {status}") } else { stderr });
            }
            Ok(None) => {
                if max_dur.is_some_and(|limit| start.elapsed() > limit) {
                    kill_process_tree(&mut child);
                    let partial_out = stdout_buf.lock().map(|g| String::from_utf8_lossy(&g).trim().to_string()).unwrap_or_default();
                    let partial_err = stderr_buf.lock().map(|g| String::from_utf8_lossy(&g).trim().to_string()).unwrap_or_default();
                    if !partial_out.is_empty() && partial_out.contains("<gridcraft_tool>") {
                        return Ok(partial_out);
                    }
                    if !partial_err.is_empty() {
                        return Err(format!("Request timed out after {timeout_secs}s: {partial_err}"));
                    }
                    return Err(format!("Request timed out after {timeout_secs}s waiting for response from `{program}`."));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_model_filter_drops_non_chat_endpoints_and_duplicates() {
        let ids = vec![
            "deepseek-chat".to_string(),
            "text-embedding-3-large".to_string(),
            "whisper-1".to_string(),
            "tts-1".to_string(),
            "omni-moderation-latest".to_string(),
            "deepseek-reasoner".to_string(),
            "deepseek-chat".to_string(),
            "  ".to_string(),
        ];
        let kept = filter_chat_models(ids);
        assert_eq!(kept, vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()]);
    }

    #[test]
    fn merge_prefers_the_provider_order_and_appends_candidates() {
        let listed = vec!["deepseek-reasoner".to_string(), "deepseek-chat".to_string()];
        let merged = merge_api_models(listed, &["deepseek-chat", "deepseek-flash"]);
        assert_eq!(merged, vec!["deepseek-reasoner", "deepseek-chat", "deepseek-flash"]);
    }

    #[test]
    fn candidate_order_tries_the_users_pick_then_discovered_then_defaults() {
        let discovered = vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()];
        let order = build_api_candidate_order(Some("deepseek-reasoner"), &discovered, &["deepseek-chat", "deepseek-flash"]);
        assert_eq!(order, vec!["deepseek-reasoner", "deepseek-chat", "deepseek-flash"]);

        let no_pick = build_api_candidate_order(None, &[], &["deepseek-chat"]);
        assert_eq!(no_pick, vec!["deepseek-chat"]);

        let blank_pick = build_api_candidate_order(Some("   "), &[], &["a", "b"]);
        assert_eq!(blank_pick, vec!["a", "b"]);
    }

    #[test]
    fn deepseek_offers_the_public_chat_models_before_the_builtin_ones() {
        let candidates = ApiProvider::DeepSeek.candidate_models();
        assert_eq!(candidates.first(), Some(&"deepseek-chat"));
        assert!(candidates.contains(&"deepseek-reasoner"));
    }

    #[test]
    fn provider_names_parse_for_control_channel_and_tests() {
        assert_eq!(ApiProvider::from_str_name("deepseek"), Some(ApiProvider::DeepSeek));
        assert_eq!(ApiProvider::from_str_name("DeepSeek"), Some(ApiProvider::DeepSeek));
        assert_eq!(ApiProvider::from_str_name(" openai "), Some(ApiProvider::OpenAI));
        assert_eq!(ApiProvider::from_str_name("claude"), Some(ApiProvider::Anthropic));
        assert_eq!(ApiProvider::from_str_name("gemini"), Some(ApiProvider::Gemini));
        assert_eq!(ApiProvider::from_str_name("openrouter"), Some(ApiProvider::OpenRouter));
        assert_eq!(ApiProvider::from_str_name("custom"), Some(ApiProvider::Custom));
        assert_eq!(ApiProvider::from_str_name(""), None);
        assert_eq!(ApiProvider::from_str_name("brand-new-provider"), None);
    }

    #[test]
    fn switching_provider_resets_the_model_dropdown_to_its_candidates() {
        let mut state = AiAgentState::default();
        state.api_models.clear();
        state.set_provider(ApiProvider::DeepSeek);
        assert_eq!(state.model, "deepseek-chat");
        assert_eq!(state.api_models.first().map(String::as_str), Some("deepseek-chat"));
        assert!(state.api_models.contains(&"deepseek-reasoner".to_string()));
        state.select_api_model("deepseek-reasoner");
        assert_eq!(state.model, "deepseek-reasoner");
    }
}
