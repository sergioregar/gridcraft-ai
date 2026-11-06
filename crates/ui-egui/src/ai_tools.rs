//! Full GridCraft access for the built-in AI Agent (`ai_agent`).
//!
//! The agent is not limited to a handful of named tools: every GridCraft feature is a command
//! (353 engine commands plus the desktop's UI-only commands), and all of them are reachable over
//! the in-process control channel with `engine.execute`. This module gives the agent what it needs
//! to use that surface:
//!
//! - [`parse_catalog`] / [`resolve_command_id`] — read the live command catalog and forgive the
//!   short names models like to use (`bold` → `home.bold`, `sortDescending` → `data.sortDescending`).
//! - [`catalog_prompt`] — a compact, grouped command index that is put in the system prompt so the
//!   model knows what exists before it acts.
//! - [`openai_tools`] — the tool schema (native function calling) for cloud APIs, Ollama and the
//!   text-tag fallback. [`FEATURE_GUIDE`] and [`CHEAT_SHEET`] carry the same knowledge to models
//!   without native tool calling (the Terminal CLI agents).
//! - [`ui_method_allowed`] — the allowlist behind `ui_control`, so UI/dialog actions are reachable
//!   without letting a model quit the app or drive blind pointer/keyboard input.
//!
//! Nothing here mutates the workbook: execution goes through `ai_agent::execute_mcp_tool`.

use serde_json::{Map, Value, json};

/// One command from `engine.commands` (engine commands plus the desktop's UI-only commands).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CommandInfo {
    pub id: String,
    pub label: String,
    pub menu: Vec<String>,
    pub params: String,
    pub shortcut: Option<String>,
    pub shipped: bool,
}

impl CommandInfo {
    /// Compact shape returned to the model by `list_commands`.
    pub(crate) fn to_compact(&self) -> Value {
        let mut params: String = self.params.chars().take(320).collect();
        if self.params.chars().count() > 320 {
            params.push('…');
        }
        json!({
            "id": self.id,
            "label": self.label,
            "params": params,
            "menu": self.menu,
            "shortcut": self.shortcut,
        })
    }

    /// Full shape returned by `describe_command` / `list_commands` lookups.
    pub(crate) fn to_full(&self) -> Value {
        json!({
            "id": self.id,
            "label": self.label,
            "params": self.params,
            "menu": self.menu,
            "shortcut": self.shortcut,
            "shipped": self.shipped,
        })
    }

    fn area(&self) -> &str {
        match self.menu.first() {
            Some(first) if !first.is_empty() => first,
            _ => "Programmatic (no ribbon)",
        }
    }
}

/// Read the JSON array `engine.commands` returns into [`CommandInfo`] values.
pub(crate) fn parse_catalog(value: &Value) -> Vec<CommandInfo> {
    let Some(rows) = value.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(id) = row.get("id").and_then(Value::as_str) else {
            continue;
        };
        if id.is_empty() {
            continue;
        }
        let label = row.get("label").and_then(Value::as_str).unwrap_or(id);
        let params = row.get("params").and_then(Value::as_str).unwrap_or("");
        let menu =
            row.get("menu").and_then(Value::as_array).map(|m| m.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
        let shortcut = row.get("shortcut").and_then(Value::as_str).map(str::to_string);
        let shipped = row.get("shipped").and_then(Value::as_bool).unwrap_or(true);
        out.push(CommandInfo { id: id.to_string(), label: label.to_string(), menu, params: params.to_string(), shortcut, shipped });
    }
    out
}

/// Lowercase and keep only ASCII letters/digits: `"Data > Sort & Filter"` and
/// `"data.sortAscending"` both become `"datasortfilter"` / `"datasortascending"`, so short names and
/// ribbon paths resolve to command ids.
fn squash(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// Resolve a model-supplied command name to an exact command id.
///
/// Tries, in order: the exact id (case-insensitive), a unique `.<name>` suffix, a squashed match
/// against the whole id (`sortAscending`, `home bold`), a squashed suffix match, an exact label
/// match, and finally a unique substring match over ids and labels. Failures return an actionable
/// message listing the candidates instead of a bare "unknown command".
pub(crate) fn resolve_command_id(catalog: &[CommandInfo], wanted: &str) -> Result<String, String> {
    let raw = wanted.trim();
    if raw.is_empty() {
        return Err(
            "missing `command`: pass a GridCraft command id, e.g. \"home.bold\" or \"data.sortAscending\". Call list_commands to discover ids."
                .into(),
        );
    }
    if let Some(hit) = catalog.iter().find(|c| c.id.eq_ignore_ascii_case(raw)) {
        return Ok(hit.id.clone());
    }

    let lower = raw.to_ascii_lowercase();
    let suffix = format!(".{lower}");
    let suffix_matches: Vec<&CommandInfo> = catalog.iter().filter(|c| c.id.to_ascii_lowercase().ends_with(&suffix)).collect();
    if suffix_matches.len() == 1 {
        return Ok(suffix_matches[0].id.clone());
    }

    let squashed = squash(raw);
    if !squashed.is_empty() {
        let exact_squashed: Vec<&CommandInfo> = catalog.iter().filter(|c| squash(&c.id) == squashed || squash(&c.label) == squashed).collect();
        if exact_squashed.len() == 1 {
            return Ok(exact_squashed[0].id.clone());
        }
        let ending: Vec<&CommandInfo> = catalog.iter().filter(|c| squash(&c.id).ends_with(&squashed)).collect();
        if ending.len() == 1 {
            return Ok(ending[0].id.clone());
        }
    }

    let label_matches: Vec<&CommandInfo> = catalog.iter().filter(|c| c.label.eq_ignore_ascii_case(raw)).collect();
    if label_matches.len() == 1 {
        return Ok(label_matches[0].id.clone());
    }

    let mut query = squashed;
    let mut hits = search_squashed(catalog, &query, 9);
    if hits.is_empty() && query.ends_with('s') {
        query.pop();
        hits = search_squashed(catalog, &query, 9);
    }
    if hits.len() == 1 {
        return Ok(hits[0].id.clone());
    }
    if hits.is_empty() {
        return Err(format!(
            "unknown command `{raw}`: no GridCraft command matches it. Call list_commands {{\"search\":\"{raw}\"}} to find the right id, or describe_command for one candidate."
        ));
    }
    let sample: Vec<&str> = hits.iter().take(8).map(|c| c.id.as_str()).collect();
    Err(format!("command `{raw}` is ambiguous; use one of these exact ids: {}. Call describe_command for its params.", sample.join(", ")))
}

fn search_squashed<'a>(catalog: &'a [CommandInfo], query: &str, limit: usize) -> Vec<&'a CommandInfo> {
    if query.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<&CommandInfo> = catalog.iter().filter(|c| squash(&c.id).contains(query) || squash(&c.label).contains(query)).collect();
    out.truncate(limit);
    out
}

/// Commands matching a model's keyword, across id, label, ribbon path and params documentation.
/// `area` optionally filters by ribbon tab (`"Home"`, `"Data"`, …).
pub(crate) fn search_commands<'a>(catalog: &'a [CommandInfo], search: &str, area: Option<&str>, limit: usize) -> Vec<&'a CommandInfo> {
    let query = squash(search);
    let area_query = area.map(squash).filter(|a| !a.is_empty());
    let mut out: Vec<&CommandInfo> = catalog
        .iter()
        .filter(|c| match &area_query {
            Some(wanted) => squash(c.area()) == *wanted,
            None => true,
        })
        .filter(|c| {
            if query.is_empty() {
                return true;
            }
            squash(&c.id).contains(&query)
                || squash(&c.label).contains(&query)
                || c.menu.iter().any(|m| squash(m).contains(&query))
                || squash(&c.params).contains(&query)
        })
        .collect();
    out.truncate(limit.max(1));
    out
}

/// Build the params object for a command from model arguments.
///
/// Accepts both shapes models produce: `{"params":{...}}` and flattened arguments
/// (`{"command":"home.bold","range":"A1:C1"}`), plus a JSON-string `params` (some local models
/// stringify it). Keys that belong to the tool call itself are not copied into the params.
pub(crate) fn merge_command_params(args: &Map<String, Value>) -> Value {
    let mut params = match args.get("params") {
        Some(Value::Object(obj)) => Value::Object(obj.clone()),
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).ok().filter(Value::is_object).unwrap_or_else(|| json!({})),
        _ => json!({}),
    };
    if let Some(obj) = params.as_object_mut() {
        for (key, value) in args {
            if matches!(
                key.as_str(),
                "command" | "command_id" | "id" | "params" | "commands" | "stopOnError" | "stop_on_error" | "search" | "area" | "limit"
            ) {
                continue;
            }
            obj.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    params
}

/// Catalog command ids that Excel has but GridCraft does not implement yet (from
/// `docs/parity-checklist.md`). Telling the model up front avoids long guessing loops.
pub(crate) const UNIMPLEMENTED_COMMANDS: &[(&str, &str)] = &[
    ("insert.smartArt", "SmartArt"),
    ("insert.screenshot", "Screenshot"),
    ("insert.map", "Maps"),
    ("insert.timeline", "Timeline"),
    ("insert.wordArt", "WordArt"),
    ("insert.signatureLine", "Signature Line"),
    ("insert.object", "Object"),
    ("insert.equation", "Equation"),
    ("data.fromWeb", "From Web"),
    ("data.stocks", "Stocks"),
    ("data.geography", "Geography"),
    ("data.relationships", "Relationships"),
    ("data.analyzeData", "Analyze Data"),
    ("review.smartLookup", "Smart Lookup"),
    ("review.translate", "Translate"),
    ("review.showChanges", "Show Changes"),
];

/// Returns the Excel feature name when `wanted` is a known-unimplemented catalog entry.
pub(crate) fn known_unimplemented(wanted: &str) -> Option<&'static str> {
    let trimmed = wanted.trim();
    UNIMPLEMENTED_COMMANDS.iter().find(|(id, _)| id.eq_ignore_ascii_case(trimmed)).map(|(_, name)| *name)
}

// ============================================================================
// Report layout geometry (pure, unit-tested): column widths from FORMATTED display
// lengths, column kinds for alignment/number formats, row roles and spacer points.
// No colours or theme are chosen here: the layout is deliberately neutral.
// ============================================================================

/// A1 letters for a 0-based column index (A, B, ..., Z, AA, ...).
pub(crate) fn column_letters(mut col: u32) -> String {
    let mut out = String::new();
    loop {
        let digit = (col % 26) as u8 + b'A';
        out.insert(0, digit as char);
        if col < 26 {
            return out;
        }
        col = col / 26 - 1;
    }
}

/// Whether a label reads like a totals/summary row ("TOTAL", "Subtotal", "Gross profit", "Net",
/// "Net", "Gross", ...). Also covers statement subtotals such as "OPERATING INCOME",
/// "INCOME BEFORE TAX", "INCOME TAX" and "NET INCOME", which close a section like a total row.
pub(crate) fn report_is_total_label(label: &str) -> bool {
    let low = label.trim().to_ascii_lowercase();
    let words = ["total", "subtotal", "sub total", "grand total", "gross", "net", "balance"];
    let starts = words.iter().any(|word| {
        low == *word || low.starts_with(&format!("{word} ")) || low.starts_with(&format!("{word}:")) || low.starts_with(&format!("{word} -"))
    });
    let statement_totals = [
        "operating income",
        "operating profit",
        "operating loss",
        "income before tax",
        "income tax",
        "profit before tax",
        "profit after tax",
        "earnings before",
        "ebit",
        "ebitda",
    ];
    starts || statement_totals.iter().any(|needle| low.contains(needle))
}

/// A label-only row (first cell has text, every other cell empty): a section heading such as
/// "REVENUE" / "COST OF GOODS SOLD".
pub(crate) fn report_is_section_row(row: &[Value]) -> bool {
    let Some(first) = row.first() else { return false };
    let has_label = first.as_str().is_some_and(|text| !text.trim().is_empty());
    if !has_label {
        return false;
    }
    !report_is_total_label(first.as_str().unwrap_or(""))
        && row.iter().skip(1).all(|cell| cell.is_null() || cell.as_str().is_some_and(|text| text.trim().is_empty()))
}

/// A cell holds a number or a formula (the two things that need a numeric format/right alignment).
pub(crate) fn report_cell_is_numeric(cell: &Value) -> bool {
    if cell.is_number() {
        return true;
    }
    match cell.as_str() {
        Some(text) => {
            let trimmed = text.trim();
            trimmed.starts_with('=') || trimmed.trim_start_matches(['$', '€', '£', '¥']).replace([',', ' ', '(', ')'], "").parse::<f64>().is_ok()
        }
        None => false,
    }
}

/// What a column holds: drives header/data alignment, number format and minimum width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReportColumnKind {
    /// Long labels (description, account, product, item name).
    Label,
    /// Short text (category, department, notes).
    Text,
    /// Money amounts (price, revenue, cost, total...).
    Money,
    /// Quantities/targets/units.
    Quantity,
    /// Percentages, variances, growth.
    Percent,
    /// Codes, dates, short statuses: centred.
    Code,
}

impl ReportColumnKind {
    pub(crate) fn header_align(self) -> &'static str {
        match self {
            ReportColumnKind::Label | ReportColumnKind::Text => "Left",
            ReportColumnKind::Money | ReportColumnKind::Quantity | ReportColumnKind::Percent => "Right",
            ReportColumnKind::Code => "Center",
        }
    }

    pub(crate) fn data_align(self) -> &'static str {
        self.header_align()
    }

    /// Minimum width in characters (before content/padding) per the layout standard.
    pub(crate) fn floor_width(self) -> u32 {
        match self {
            ReportColumnKind::Label => 38,
            ReportColumnKind::Text => 18,
            ReportColumnKind::Money => 18,
            ReportColumnKind::Quantity => 14,
            ReportColumnKind::Percent => 14,
            ReportColumnKind::Code => 16,
        }
    }

    /// Number format for the data cells. Money is accounting (no forced currency symbol) and only
    /// shows decimals when the column really has fractional cents; integer values never show `.00`.
    pub(crate) fn number_format(self, has_cents: bool) -> &'static str {
        match self {
            ReportColumnKind::Money | ReportColumnKind::Quantity => {
                if has_cents {
                    "#,##0.00_);(#,##0.00);\"-\""
                } else {
                    "#,##0_);(#,##0);\"-\""
                }
            }
            ReportColumnKind::Percent => "+0.0%;-0.0%;0.0%",
            ReportColumnKind::Label | ReportColumnKind::Text | ReportColumnKind::Code => "General",
        }
    }
}

fn header_words(header: &str) -> String {
    header.to_ascii_lowercase()
}

/// Classify a column from its header, falling back to the data values.
pub(crate) fn classify_report_column(header: &str, values: &[Value]) -> ReportColumnKind {
    let low = header_words(header);
    let has = |needles: &[&str]| needles.iter().any(|needle| low.contains(needle));
    if has(&["%", "percent", "persen", "variance", "varians", "growth", "pertumbuhan", "margin", "rate", "yoy"]) {
        return ReportColumnKind::Percent;
    }
    if has(&["id", "code", "no.", "number", "date", "status", "sku"]) {
        return ReportColumnKind::Code;
    }
    if has(&["qty", "quantity", "unit", "units", "target", "sold", "count", "stock", "pcs"]) {
        return ReportColumnKind::Quantity;
    }
    if has(&[
        "price",
        "price",
        "amount",
        "amount",
        "total",
        "revenue",
        "pendapatan",
        "sales",
        "sales",
        "cost",
        "cogs",
        "hpp",
        "expense",
        "beban",
        "biaya",
        "income",
        "profit",
        "loss",
        "profit",
        "gaji",
        "salary",
        "bonus",
        "balance",
        "saldo",
        "debit",
        "credit",
        "kredit",
        "asset",
        "aset",
        "liability",
        "subtotal",
        "gross",
        "net",
        "rp",
        "usd",
        "$",
    ]) {
        return ReportColumnKind::Money;
    }
    if has(&["category", "department", "type", "group", "class", "segment"]) {
        return ReportColumnKind::Text;
    }
    if has(&["description", "account", "product", "item", "name", "detail", "label", "title"]) {
        return ReportColumnKind::Label;
    }
    if values.iter().any(report_cell_is_numeric) {
        return ReportColumnKind::Money;
    }
    ReportColumnKind::Text
}

/// Number of decimal cents that actually occur in a money column (0 unless a value is fractional).
pub(crate) fn report_money_has_cents(values: &[Value]) -> bool {
    values.iter().any(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str().and_then(|text| text.trim().replace([',', ' '], "").parse::<f64>().ok()))
            .is_some_and(|number| (number.fract().abs() * 100.0).round() >= 1.0)
    })
}

fn group_integer(digits: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = digits.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if index > 0 && (chars.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*ch);
    }
    out
}

fn format_fixed(number: f64, decimals: usize, sign: bool) -> String {
    let negative = number.is_sign_negative() && number.abs() >= 0.5 / 10_f64.powi(decimals as i32);
    let abs = number.abs();
    let factor = 10_f64.powi(decimals as i32);
    let scaled = (abs * factor).round().max(0.0) as u128;
    let integer = scaled / (factor as u128);
    let fraction = scaled % (factor as u128);
    let mut out = String::new();
    if negative {
        out.push('-');
    } else if sign {
        out.push('+');
    }
    out.push_str(&group_integer(&integer.to_string()));
    if decimals > 0 {
        out.push('.');
        out.push_str(&format!("{fraction:0width$}", width = decimals));
    }
    out
}

/// Display length of a cell after its number format is applied (this is what must fit the column,
/// not the raw value). `LayoutColumnKind` decides grouping/percent handling.
pub(crate) fn report_formatted_length(value: &Value, kind: ReportColumnKind, has_cents: bool) -> usize {
    let number = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return 0;
            }
            if matches!(kind, ReportColumnKind::Label | ReportColumnKind::Text | ReportColumnKind::Code) {
                return trimmed.chars().count();
            }
            trimmed.trim_start_matches(['$', '€', '£', '¥', '(', '-']).replace([',', ' ', ')'], "").parse::<f64>().ok()
        }
        _ => None,
    };
    let Some(number) = number else {
        return if matches!(kind, ReportColumnKind::Label | ReportColumnKind::Text | ReportColumnKind::Code) {
            value.as_str().map(|text| text.chars().count()).unwrap_or(0)
        } else {
            0
        };
    };
    match kind {
        ReportColumnKind::Money | ReportColumnKind::Quantity => {
            let decimals = if matches!(kind, ReportColumnKind::Money) && has_cents { 2 } else { 0 };
            let text = format_fixed(number, decimals, false);
            // Accounting format adds a trailing padding space for the `_)` token.
            text.chars().count() + 1
        }
        ReportColumnKind::Percent => format_fixed(number * 100.0, 1, true).chars().count() + 1,
        ReportColumnKind::Label | ReportColumnKind::Text | ReportColumnKind::Code => {
            let text = if number.fract() == 0.0 { format_fixed(number, 0, false) } else { format_fixed(number, 2, false) };
            text.chars().count()
        }
    }
}

/// Width in characters for one column: max(formatted content, header) + safe padding (+6), with the
/// kind's minimum floor. This guarantees no `###` and a visible whitespace gap before the next column.
pub(crate) fn report_column_width(header: &str, values: &[Value], kind: ReportColumnKind) -> u32 {
    let has_cents = matches!(kind, ReportColumnKind::Money) && report_money_has_cents(values);
    let content = values.iter().map(|value| report_formatted_length(value, kind, has_cents)).max().unwrap_or(0);
    let header_len = header.chars().count();
    let width = content.max(header_len) as u32 + 6;
    width.max(kind.floor_width()).min(120)
}

/// Row role inside a structured/financial report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReportRowRole {
    /// "REVENUE", "COST OF GOODS SOLD": bold, left, no values.
    Section,
    /// A line item under a section: regular, left, indented when inside a section.
    LineItem,
    /// "TOTAL REVENUE", "GROSS PROFIT": bold, top+bottom thin borders.
    Subtotal,
    /// The bottom-most total ("NET INCOME"): bold, top thin + bottom double.
    GrandTotal,
}

/// Roles for the data rows: section headings are their own level; the bottom-most total becomes the
/// grand total, earlier totals are section subtotals, everything else is a line item.
pub(crate) fn report_row_roles(labels: &[Option<String>], sections: &[bool]) -> Vec<ReportRowRole> {
    let total_flags: Vec<bool> = labels.iter().map(|label| label.as_deref().is_some_and(report_is_total_label)).collect();
    let last_total = total_flags.iter().rposition(|flag| *flag);
    labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            if sections.get(index).copied().unwrap_or(false) {
                return ReportRowRole::Section;
            }
            let Some(label) = label else { return ReportRowRole::LineItem };
            if label.trim().is_empty() {
                return ReportRowRole::LineItem;
            }
            if total_flags[index] {
                if Some(index) == last_total { ReportRowRole::GrandTotal } else { ReportRowRole::Subtotal }
            } else {
                ReportRowRole::LineItem
            }
        })
        .collect()
}

/// A row is a section heading when the label is present but no value cells follow it.
pub(crate) fn report_section_flags(rows: &[Vec<Value>]) -> Vec<bool> {
    rows.iter().map(|row| report_is_section_row(row)).collect()
}

/// Marks input rows after which a 10pt blank spacer must be inserted: a total/subtotal row that is
/// followed by a new section heading. Exactly one blank row per section boundary.
pub(crate) fn report_spacer_after(roles: &[ReportRowRole], sections: &[bool]) -> Vec<bool> {
    let mut spacers = vec![false; roles.len()];
    for index in 0..roles.len().saturating_sub(1) {
        let closes_section = matches!(roles[index], ReportRowRole::Subtotal | ReportRowRole::GrandTotal);
        if closes_section && sections[index + 1] {
            spacers[index] = true;
        }
    }
    spacers
}

/// Highest-priority role definition for every backend (cloud API, Terminal CLI and local Ollama).
/// The pane is a workbook operator, not a chat assistant: user messages are work orders, and the
/// agent acts on the live workbook instead of answering questions.
/// Highest-priority role definition for every backend (cloud API, Terminal CLI and local Ollama).
/// Deliberately compact (the prompt is kept light so small local models can follow it) and
/// intent-first: work orders are executed, questions are answered from tool results, out-of-topic
/// questions are declined, and an empty workbook is filled by generating the requested data.
pub(crate) const GROUND_RULES: &str = "\
=== ROLE AND GROUND RULES (highest priority) ===
You are the GridCraft Workbook Agent: part operator, part assistant, inside a live spreadsheet.
Decide the intent of every message first:
1. WORK ORDER (create, edit, format, sort, filter, delete, chart, pivot, export, ...) -> perform it
   with the tools on the live workbook, then report what you did in at most 3 short lines.
2. QUESTION ABOUT THE WORKBOOK (\"how many rows?\", \"best seller?\", \"what is the total?\") -> inspect
   or compute it with tools. Answer briefly from the tool results (and write the result into cells
   when that helps the user).
3. QUESTION ABOUT GRIDCRAFT OR SPREADSHEETS (\"how do I freeze panes?\") -> answer in 1-3 lines and
   offer to do it; do not write a tutorial.
4. OUT-OF-TOPIC question (news, general knowledge, coding help, chit-chat) -> decline in one short
   line (\"I only work on this workbook.\") and do nothing. Never answer it.
5. EMPTY WORKBOOK + a request to make a report, list or data -> CREATE the data yourself:
   write a title row, a header row, at least 10 plausible rows (unique product names or categories,
   quantities, unit prices, dates), formulas (quantity*price, a totals row), number formats, AutoFit the
   columns for readability, a clean table style, and a chart when asked. This is allowed: \"never invent data\" means never misreport
   the clean layout (merged title, readable widths, formats, totals) in one call. Never leave ### or
   truncated labels in a finished sheet.
Rules for every action: read before writing; prefer one focused tool call or a small
execute_commands batch; verify structural edits with read_range/inspect_workbook; never fabricate
command ids or results; if a command fails, read the error and retry once, then say what failed.
Ask at most one short clarifying question only when the intent cannot be determined.
Always reply in English, whatever language the user writes in.
";

/// The system-prompt section that explains how to reach every feature. Kept compact on purpose.
pub(crate) const FEATURE_GUIDE: &str = "\
=== FULL FEATURE ACCESS (every feature is a command) ===
Discover with list_commands {\"search\":\"sort\"} (no search = whole catalog), describe_command
{\"command\":\"id\"} for params, run one with execute_command {\"command\":\"id\",\"params\":{...}} (short
ids work: bold -> home.bold), batch with execute_commands {\"commands\":[...]}, list_functions for the
514 worksheet functions. ui_control {\"method\":...,\"params\":...} reaches UI-only actions (ui.inspect,
ui.ribbon, ui.dialog, app.language.*). Commands act on the ACTIVE sheet: pass range/sheet explicitly.
Always finish table/report work with the layout pass: create_report for a new report, layout_sheet to
repair an existing range. Never claim a feature is missing before list_commands finds no match.
";

/// Prompt-ready recipes, condensed to keep the prompt light. Every example is a command this build
/// implements; params are documented by `describe_command`.
pub(crate) const CHEAT_SHEET: &str = "\
=== RECIPES, LAYOUT RULES AND COMMANDS (validated) ===
FINAL LAYOUT PASS: before finishing any table/report creation or edit, run create_report (new report)
or layout_sheet (existing range). That pass comes AFTER all values and formulas are written; never
leave the sheet half-formatted.
ATURAN KERAPIAN MUTLAK: (1) never leave ###, clipped text or values touching the next column: width
columns from the FORMATTED display length plus padding 6 (label 36-42, number 16-20, quantity and
percent 12-14 characters). (2) merged title/subtitle are left or centre aligned (never right); title
bold 14-16pt, subtitle italic 10-11pt, no box border, exactly ONE blank 10pt spacer row before the
column header row. (3) header alignment matches its column and stays straight vertically: text left,
number/currency/percent right, code/date/status centre; vertical alignment centre everywhere; never
wrap short headers, widen the column instead. (4) sectioned reports: section headings bold left with
indent 0; line items regular left with indent 1; subtotal rows bold left with top+bottom thin
borders; the grand total bold left with top thin + bottom DOUBLE border; after a subtotal/grand
total that is followed by a new section insert exactly one blank 10pt spacer row, and no other free
spacer rows. (5) NO full grid box borders and no vertical borders: header row top thin + bottom
medium; section rows no border; line items no border; subtotal top+bottom thin; grand total top thin
+ bottom double. (6) whole money/quantity values show no .00 (format #,##0_);(#,##0);\"-\"), percent
uses one decimal (+0.0%;-0.0%;0.0%) and is right aligned; row heights: title 27, subtitle 19, spacer
10, header 25, section 22, line item 19, subtotal 22, grand total 24pt; freeze below the header row;
print fit to 1 page wide. Keep colours neutral: no fills, font colours or table styles.
create_report {title?,subtitle?,headers:[...],rows:[[label,value,...]],chart?} builds
all of it in one call; layout_sheet {range?,header_row?}
repairs an existing range without changing values. Executors: write_range, read_range, execute_command
and execute_commands (batch), list_commands, describe_command, list_functions, ui_control. Commands:
cell.set/range.setValues; home.* formatting; data.* sort/filter/validation; insert.chart/table/
pivotTable; view.freezePanes; pageLayout.printArea/pageSetup; file.saveAs/exportPdf.
Compact ids: write_range/read_range; home.formatCells, home.bold, home.numberFormat, home.columnWidth, home.rowHeight,
home.borders, home.mergeCells, home.alignLeft|Center|Right|Middle, home.increaseIndent, home.wrapText;
data.sortAscending, data.sortDescending, data.filterBy, data.validation; insert.table, insert.chart, insert.pivotTable,
insert.sparkline; view.freezePanes, view.zoom; pageLayout.printArea, pageLayout.pageSetup; file.saveAs, file.exportPdf.
Newer features are reachable too: stacked100 chart subtype; pageLayout.headerFooter scaleWithDoc/alignWithMargins;
data.outlineSettings; edit.fillSeries dateUnit; pivot.options enableShowDetails; formulas.errorChecking;
app.checkForUpdates; ui.dialog commandSearch/agents/journal; view.theme; view.interfaceScale.
";

/// Whitelisted control-channel methods for the `ui_control` tool. Everything else (quit, printing,
/// pointer/keyboard injection, dialogs' OK button) stays out of reach: commands cover those actions
/// with explicit params, where the model states what it wants instead of clicking a UI.
pub(crate) const UI_METHODS: &[&str] = &[
    "ui.inspect",
    "ui.ribbon",
    "ui.set",
    "ui.message.dismiss",
    "ui.dialog",
    "ui.dialog.set",
    "ui.dialog.cancel",
    "app.language.set",
    "app.language.english",
    "app.language.japanese",
    "engine.journal",
    "selection.stats",
    "document.inspect",
    "sheet.read",
    "cell.get",
];

/// Whether `method` may be reached through `ui_control`.
pub(crate) fn ui_method_allowed(method: &str) -> bool {
    UI_METHODS.contains(&method)
}

/// Names of the tools exposed for an agent configuration; one source for the schema, the Terminal
/// prompt and the panel summary.
pub(crate) fn tool_names(full_access: bool, allow_ui: bool) -> Vec<&'static str> {
    let mut names = vec![
        "inspect_workbook",
        "read_range",
        "write_range",
        "set_cell",
        "get_cell",
        "format_range",
        "create_table",
        "insert_chart",
        "clear_range",
        "delete_chart",
        "evaluate_formula",
        "list_commands",
        "execute_command",
    ];
    if full_access {
        names.extend([
            "describe_command",
            "execute_commands",
            "list_functions",
            "select",
            "sort_range",
            "create_report",
            "layout_sheet",
            "undo",
            "redo",
        ]);
        if allow_ui {
            names.push("ui_control");
        }
    }
    names
}

/// Native function-calling schema. `full_access` adds the catalog-aware tools; `allow_ui` adds
/// `ui_control`.
pub(crate) fn openai_tools(full_access: bool, allow_ui: bool) -> Value {
    let mut tools: Vec<Value> = vec![
        json!({
            "type": "function",
            "function": {
                "name": "inspect_workbook",
                "description": "Summary of the active workbook: title, sheets, used ranges, tables, charts, defined names, current selection and active cell.",
                "parameters": {"type": "object", "properties": {}}
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_range",
                "description": "Read cell values from an A1 range (e.g. 'A1:D10' or 'Sheet1!A1:C5') as a 2-D array. Omit range to read the used range. Read modest ranges (a few hundred rows) instead of whole columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string", "description": "A1-style range, e.g. 'A1:D10'"},
                        "sheet": {"type": "string", "description": "Optional sheet name"},
                        "formulas": {"type": "boolean", "description": "Return formulas instead of values"},
                        "formatted": {"type": "boolean", "description": "Return formatted display strings"}
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_range",
                "description": "Write a 2-D array of values/formulas starting at the top-left cell of `range` (e.g. 'A1'). Strings starting with '=' are formulas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string", "description": "Top-left cell or range, e.g. 'A1'"},
                        "values": {
                            "type": "array",
                            "description": "2-D array of rows",
                            "items": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}}
                        }
                    },
                    "required": ["range", "values"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "set_cell",
                "description": "Set one cell value or formula (e.g. cell='B2', input='=SUM(A1:A10)' or 42).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {"type": "string", "description": "Cell address, e.g. 'B2'"},
                        "input": {"type": ["string", "number", "boolean"], "description": "Value or =formula"}
                    },
                    "required": ["cell", "input"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "get_cell",
                "description": "Read full details of one cell (value, formula, display text, style).",
                "parameters": {
                    "type": "object",
                    "properties": {"cell": {"type": "string"}},
                    "required": ["cell"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "format_range",
                "description": "Common formatting in one call: bold, italic, underline, fontName, fontSize, fontColor ('#RRGGBB'), fillColor ('#RRGGBB'), numberFormat ('Currency', '#,##0.00', '0%'), horizontalAlign ('left'|'center'|'right'), wrapText. For borders, protection or advanced style use execute_command home.formatCells {range,style:{...}} or home.borders.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string"},
                        "bold": {"type": "boolean"},
                        "italic": {"type": "boolean"},
                        "underline": {"type": "boolean"},
                        "fontName": {"type": "string"},
                        "fontSize": {"type": "number"},
                        "fontColor": {"type": "string"},
                        "fillColor": {"type": "string"},
                        "numberFormat": {"type": "string"},
                        "horizontalAlign": {"type": "string", "enum": ["left", "center", "right", "justify"]},
                        "wrapText": {"type": "boolean"}
                    },
                    "required": ["range"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "create_table",
                "description": "Format a range as a styled table with headers and filter buttons.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string"},
                        "name": {"type": "string"},
                        "style": {"type": "string"},
                        "header": {"type": "boolean"}
                    },
                    "required": ["range"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "insert_chart",
                "description": "Insert a chart from a range. Types: column, bar, line, pie, doughnut, area, scatter, bubble, radar, combo, histogram, waterfall, funnel, treemap, sunburst, boxWhisker, stock. Range may be multi-area ('B4:B10,G4:G10'). For full control use execute_command insert.chart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string", "description": "Source data range including headers (e.g. 'B4:F10' or 'B4:B10,G4:G10')"},
                        "type": {"type": "string", "description": "Chart type: column, bar, line, pie, doughnut, area, scatter, bubble, radar, combo, histogram, waterfall, funnel, treemap, sunburst, boxWhisker, stock"},
                        "subtype": {"type": "string", "description": "Optional subtype: clustered, stacked, stacked100, markers"},
                        "title": {"type": "string", "description": "Descriptive chart title"},
                        "at": {"type": "string", "description": "Top-left anchor cell (e.g. 'K2')"},
                        "width": {"type": "number"},
                        "height": {"type": "number"}
                    },
                    "required": ["range", "type"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "clear_range",
                "description": "Delete cell contents, backgrounds, formal tables, conditional formats, sparklines and chart widgets. Pass range for one area or all=true for the entire active sheet. Prefer the narrower edit.clearContents/edit.clearFormats commands when you only want to clear data or formats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string", "description": "A1 range to clear"},
                        "all": {"type": "boolean", "description": "True clears the entire active sheet"}
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "delete_chart",
                "description": "Delete one chart widget (chart=id) or all chart widgets on the active sheet (all=true).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart": {"type": "integer", "description": "Chart id; omit with all=true to delete every chart"},
                        "all": {"type": "boolean"}
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "evaluate_formula",
                "description": "Evaluate a formula without modifying the sheet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "formula": {"type": "string"},
                        "cell": {"type": "string"}
                    },
                    "required": ["formula"]
                }
            }
        }),
    ];

    if full_access {
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "list_commands",
                "description": "Search GridCraft's whole command surface (every ribbon/menu feature). Matches id, label, ribbon path and params. Call this BEFORE saying a feature is unavailable, then run the id with execute_command. With no search it lists every command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string", "description": "Keyword, e.g. 'sort', 'conditional', 'pivot', 'validation', 'freeze', 'page setup', 'border', 'protect'"},
                        "area": {"type": "string", "description": "Optional ribbon tab filter: File, Home, Insert, Draw, Page Layout, Formulas, Data, Review, View, Automate, Table, PivotTable Analyze, Chart Design, ..."},
                        "limit": {"type": "integer", "description": "Max commands to return (default 200)"}
                    }
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "describe_command",
                "description": "Full documentation (params, ribbon path, shortcut) of one command id. Accepts short names too ('bold', 'sortDescending').",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Run ANY GridCraft command by id with JSON params. Short names work ('bold' -> home.bold). Examples: {\"command\":\"data.sortAscending\",\"params\":{\"range\":\"A1:C10\",\"column\":\"B\"}}, {\"command\":\"view.freezePanes\",\"params\":{\"cell\":\"B2\"}}, {\"command\":\"file.exportPdf\",\"params\":{\"path\":\"report.pdf\"}}, {\"command\":\"home.conditionalFormat\",\"params\":{\"range\":\"C2:C20\",\"rule\":{\"type\":\"dataBar\"}}}. Params may also be passed directly beside \"command\".",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command id or short name"},
                        "params": {"type": "object", "description": "Command parameters (see list_commands/describe_command)"}
                    },
                    "required": ["command"]
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "execute_commands",
                "description": "Run a batch of GridCraft commands in order and get one result per command. Use for multi-step work (formatting a report, building a table then chart then formatting, resetting a view). Example: {\"commands\":[{\"command\":\"home.bold\",\"params\":{\"range\":\"A1:F1\"}},{\"command\":\"home.fillColor\",\"params\":{\"range\":\"A1:F1\",\"color\":\"#1F9D55\"}}]}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "params": {"type": "object"}
                                },
                                "required": ["command"]
                            }
                        },
                        "stopOnError": {"type": "boolean", "description": "Stop at the first failing command (default false)"}
                    },
                    "required": ["commands"]
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "list_functions",
                "description": "List/search the 514 worksheet functions with category, signature and description (e.g. search 'XLOOKUP' or category 'Financial').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "select",
                "description": "Set the selection: {\"range\":\"A1:C10\"} or {\"cell\":\"B2\"} or {\"sheet\":\"Sheet2\"}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string"},
                        "cell": {"type": "string"},
                        "sheet": {"type": "string"}
                    }
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "sort_range",
                "description": "Sort a range: {range,column:'B',order:'asc'|'desc'} or {range,keys:[{column:'B',order:'desc'}]} (multiple keys).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string"},
                        "column": {"type": "string"},
                        "order": {"type": "string", "enum": ["asc", "desc"]},
                        "keys": {"type": "array", "items": {"type": "object"}},
                        "header": {"type": "boolean"}
                    }
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "undo",
                "description": "Undo the last command step(s).",
                "parameters": {"type": "object", "properties": {"steps": {"type": "integer"}}}
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "redo",
                "description": "Redo undone step(s).",
                "parameters": {"type": "object", "properties": {"steps": {"type": "integer"}}}
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "create_report",
                "description": "Build and lay out a clean neutral table/report from scratch in one FINAL LAYOUT PASS: merged left title/subtitle (no border) with one blank 10pt spacer before the header, formatted-width columns (no ###, no clipped labels), bold header with top thin + bottom medium rules, section headings, indented line items, subtotal rules, a grand total with a double bottom rule, number formats without forced colours, frozen header and print fit-to-one-page. Use this for any 'make a report/statement/list' request instead of writing cells one by one. Rows are arrays whose first cell is the label; label-only rows are section headings and rows labelled TOTAL/Subtotal/Gross/Net/Operating income/Income before tax/Income tax/Profit/Loss are totals (the bottom-most is the grand total).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Report title, merged across all columns on row 1"},
                        "subtitle": {"type": "string", "description": "Optional italic subtitle on row 2"},
                        "headers": {"type": "array", "items": {"type": "string"}, "description": "Header row, e.g. [\"Product\",\"Category\",\"Qty\",\"Price\",\"Total\"]"},
                        "rows": {
                            "type": "array",
                            "description": "Data rows, first cell is the label",
                            "items": {"type": "array", "items": {"type": ["string", "number", "null"]}}
                        },
                        "chart": {"type": "object", "description": "Optional {\"type\":\"column|bar|line|pie\",\"title\":\"...\",\"at\":\"E2\"}"},
                        "freeze_header": {"type": "boolean", "description": "Freeze the title/header rows (default true)"},
                        "print_setup": {"type": "boolean", "description": "Set print area, orientation and fit-to-page (default true)"}
                    },
                    "required": ["headers", "rows"]
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "layout_sheet",
                "description": "Apply the same neutral FINAL LAYOUT PASS to an existing range/report WITHOUT changing any value or formula: format-driven widths (no ###), title/subtitle merge and alignment, one blank 10pt spacer below the subtitle and between sections (inserted with row shifts, so formulas stay correct), row heights, header/data alignment straight down each column, indent hierarchy, number formats, functional horizontal borders (header rule, subtotal rules, grand total double rule; no grid box), frozen header and print fit-to-one-page. Call this when a sheet looks cramped, text is clipped or numbers show ###.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string", "description": "A1 range to lay out; defaults to the used range"},
                        "header_row": {"type": "integer", "description": "1-based header row (auto-detected when omitted)"}
                    }
                }
            }
        }));
        if allow_ui {
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": "ui_control",
                    "description": "Reach UI-only actions the command catalog cannot express. Methods: ui.inspect; ui.ribbon {tab}; ui.set {dark,formulaBar,ribbonCollapsed,animations}; ui.message.dismiss; ui.dialog {name} + ui.dialog.set {field,value} + ui.dialog.cancel (dialogs cannot be confirmed here - pass full params to the equivalent command instead); app.language.set {language: code|name}; app.language.english; app.language.japanese; engine.journal; selection.stats; document.inspect; sheet.read {range,sheet}; cell.get {cell}. Read-only methods are safe at any time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "description": "One of the allowed methods"},
                            "params": {"type": "object"}
                        },
                        "required": ["method"]
                    }
                }
            }));
        }
    } else {
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "list_commands",
                "description": "List available GridCraft commands (search by keyword).",
                "parameters": {
                    "type": "object",
                    "properties": {"search": {"type": "string"}}
                }
            }
        }));
        tools.push(json!({
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Run a GridCraft engine/UI command by id with JSON params.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "params": {"type": "object"}
                    },
                    "required": ["command"]
                }
            }
        }));
    }

    Value::Array(tools)
}

/// Compact, grouped command index for the system prompt. Keeps ids and labels (enough for the model
/// to know that a feature exists) and leaves params to `describe_command`, so the prompt stays
/// affordable even with 360+ commands.
pub(crate) fn catalog_prompt(catalog: &[CommandInfo], max_chars: usize) -> String {
    if catalog.is_empty() {
        return String::new();
    }
    const AREA_ORDER: &[&str] = &[
        "File",
        "Home",
        "Insert",
        "Draw",
        "Page Layout",
        "Formulas",
        "Data",
        "Review",
        "View",
        "Automate",
        "Table",
        "Table Design",
        "PivotTable Analyze",
        "PivotTable Design",
        "Chart Design",
        "Format",
        "Edit",
        "Window",
    ];
    let mut areas: Vec<&str> = Vec::new();
    for wanted in AREA_ORDER {
        if catalog.iter().any(|c| c.area() == *wanted) {
            areas.push(wanted);
        }
    }
    if catalog.iter().any(|c| !AREA_ORDER.contains(&c.area())) {
        areas.push("Programmatic (no ribbon)");
    }

    let mut out = String::new();
    out.push_str(&format!("=== LIVE GRIDCRAFT COMMAND CATALOG ({} commands; run any id with execute_command) ===\n", catalog.len()));
    let mut truncated = false;
    'areas: for area in areas {
        let header = format!("[{area}] ");
        if out.chars().count() + header.chars().count() > max_chars {
            truncated = true;
            break;
        }
        out.push_str(&header);
        for command in catalog.iter().filter(|c| c.area() == area) {
            let entry = format!("{} \"{}\" | ", command.id, command.label);
            if out.chars().count() + entry.chars().count() > max_chars {
                truncated = true;
                break 'areas;
            }
            out.push_str(&entry);
        }
        out.push('\n');
    }
    let shown = out.matches(" | ").count();
    if truncated || shown < catalog.len() {
        out.push_str(&format!(
            "(index truncated: {} of {} commands shown; call list_commands {{\"search\":\"...\"}} for the rest)\n",
            shown,
            catalog.len()
        ));
    }
    out.push_str("Use list_commands for full params, describe_command for one command, execute_command to run it.\n");
    out
}

/// Truncate a tool result before it goes back into the conversation, so a huge range cannot blow the
/// context window of the next step.
pub(crate) fn truncate_tool_output(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}… (truncated; read a smaller range if you need the rest)")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> Vec<CommandInfo> {
        parse_catalog(&json!([
            {"id": "home.bold", "label": "Bold", "menu": ["Home", "Font"], "params": "{range?, on?: bool}", "shortcut": "Ctrl+B"},
            {"id": "home.italic", "label": "Italic", "menu": ["Home", "Font"], "params": "{range?, on?: bool}"},
            {"id": "data.sortAscending", "label": "Sort A to Z", "menu": ["Data", "Sort & Filter"], "params": "{range?, column?: \"B\"}"},
            {"id": "data.sortDescending", "label": "Sort Z to A", "menu": ["Data", "Sort & Filter"], "params": "{range?, column?}"},
            {"id": "data.sort", "label": "Sort…", "menu": ["Data", "Sort & Filter"], "params": "{range?, header?: bool, keys: [{column, order}]}"},
            {"id": "view.freezePanes", "label": "Freeze Panes", "menu": ["View", "Window"], "params": "{sheet?, cell?}"},
            {"id": "file.exportPdf", "label": "Export as PDF", "menu": ["File"], "params": "{path?, sheets?, range?}"},
            {"id": "ui.ribbonTab", "label": "Ribbon Tab", "menu": [], "params": "", "ui": true},
        ]))
    }

    #[test]
    fn resolves_exact_and_case_insensitive_ids() {
        let catalog = sample_catalog();
        assert_eq!(resolve_command_id(&catalog, "home.bold").unwrap(), "home.bold");
        assert_eq!(resolve_command_id(&catalog, "HOME.BOLD").unwrap(), "home.bold");
    }

    #[test]
    fn resolves_short_suffix_names() {
        let catalog = sample_catalog();
        assert_eq!(resolve_command_id(&catalog, "bold").unwrap(), "home.bold");
        assert_eq!(resolve_command_id(&catalog, "freezePanes").unwrap(), "view.freezePanes");
    }

    #[test]
    fn resolves_squashed_ids_and_ribbon_paths() {
        let catalog = sample_catalog();
        assert_eq!(resolve_command_id(&catalog, "sortAscending").unwrap(), "data.sortAscending");
        assert_eq!(resolve_command_id(&catalog, "Home > Bold").unwrap(), "home.bold");
        assert_eq!(resolve_command_id(&catalog, "Freeze Panes").unwrap(), "view.freezePanes");
    }

    #[test]
    fn resolves_unique_label_matches() {
        let catalog = sample_catalog();
        assert_eq!(resolve_command_id(&catalog, "Bold").unwrap(), "home.bold");
        assert_eq!(resolve_command_id(&catalog, "export as pdf").unwrap(), "file.exportPdf");
    }

    #[test]
    fn short_names_prefer_the_single_command_they_name() {
        let catalog = sample_catalog();
        // "sort" only ends one id (`data.sort`), so it resolves there instead of guessing a variant.
        assert_eq!(resolve_command_id(&catalog, "sort").unwrap(), "data.sort");
    }

    #[test]
    fn ambiguous_names_report_candidates_instead_of_guessing() {
        let catalog = sample_catalog();
        let err = resolve_command_id(&catalog, "data").unwrap_err();
        assert!(err.contains("data.sort"), "message: {err}");
        assert!(err.contains("ambiguous"), "message: {err}");
    }

    #[test]
    fn unknown_names_suggest_list_commands() {
        let catalog = sample_catalog();
        let err = resolve_command_id(&catalog, "teleport").unwrap_err();
        assert!(err.contains("list_commands"), "message: {err}");
        assert!(err.contains("teleport"), "message: {err}");
    }

    #[test]
    fn empty_name_is_rejected_with_hint() {
        let catalog = sample_catalog();
        let err = resolve_command_id(&catalog, "  ").unwrap_err();
        assert!(err.contains("list_commands"));
    }

    #[test]
    fn search_matches_params_and_area() {
        let catalog = sample_catalog();
        let hits = search_commands(&catalog, "keys", None, 20);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "data.sort");
        let home = search_commands(&catalog, "", Some("Home"), 20);
        assert_eq!(home.len(), 2);
        let programmatic = search_commands(&catalog, "ribbon", None, 20);
        assert_eq!(programmatic.len(), 1);
        assert_eq!(programmatic[0].id, "ui.ribbonTab");
    }

    #[test]
    fn params_merge_accepts_nested_flat_and_stringified_shapes() {
        let nested: Map<String, Value> = serde_json::from_value(json!({
            "command": "home.bold",
            "params": {"range": "A1:C1", "on": true}
        }))
        .unwrap();
        assert_eq!(merge_command_params(&nested), json!({"range": "A1:C1", "on": true}));

        let flat: Map<String, Value> = serde_json::from_value(json!({
            "command": "home.bold",
            "range": "A1:C1",
            "on": false
        }))
        .unwrap();
        assert_eq!(merge_command_params(&flat), json!({"range": "A1:C1", "on": false}));

        let stringified: Map<String, Value> = serde_json::from_value(json!({
            "command": "home.bold",
            "params": "{\"range\":\"A1:C1\"}"
        }))
        .unwrap();
        assert_eq!(merge_command_params(&stringified), json!({"range": "A1:C1"}));

        let malformed: Map<String, Value> = serde_json::from_value(json!({
            "command": "home.bold",
            "params": "{not json}"
        }))
        .unwrap();
        // A malformed string falls back to an empty object instead of failing the tool call.
        assert_eq!(merge_command_params(&malformed), json!({}));

        let mut nested_wins: Map<String, Value> = serde_json::from_value(json!({
            "command": "x",
            "range": "A1",
            "params": {"range": "B2"}
        }))
        .unwrap();
        assert_eq!(merge_command_params(&nested_wins), json!({"range": "B2"}));
        nested_wins.remove("params");
        assert_eq!(merge_command_params(&nested_wins), json!({"range": "A1"}));
    }

    #[test]
    fn ui_allowlist_keeps_quit_print_and_pointer_out() {
        assert!(ui_method_allowed("ui.inspect"));
        assert!(ui_method_allowed("ui.ribbon"));
        assert!(ui_method_allowed("document.inspect"));
        assert!(!ui_method_allowed("app.quit"));
        assert!(!ui_method_allowed("ui.printSubmit"));
        assert!(!ui_method_allowed("ui.dialog.confirm"));
        assert!(!ui_method_allowed("ui.key"));
        assert!(!ui_method_allowed("ui.click"));
        assert!(!ui_method_allowed("app.open"));
    }

    #[test]
    fn known_unimplemented_features_are_recognised() {
        assert_eq!(known_unimplemented("insert.smartArt"), Some("SmartArt"));
        assert_eq!(known_unimplemented("DATA.STOCKS"), Some("Stocks"));
        assert_eq!(known_unimplemented("home.bold"), None);
    }

    #[test]
    fn full_tool_access_exposes_batch_and_ui_tools() {
        let full = tool_names(true, true);
        assert!(full.contains(&"execute_commands"));
        assert!(full.contains(&"describe_command"));
        assert!(full.contains(&"ui_control"));
        let full_no_ui = tool_names(true, false);
        assert!(full_no_ui.contains(&"execute_commands"));
        assert!(!full_no_ui.contains(&"ui_control"));
        let legacy = tool_names(false, true);
        assert_eq!(legacy.len(), 13);
        assert!(!legacy.contains(&"execute_commands"));
        assert!(!legacy.contains(&"ui_control"));
    }

    #[test]
    fn tool_schema_follows_the_access_flags() {
        let full = openai_tools(true, true);
        let names = tool_schema_names(&full);
        assert!(names.contains(&"execute_commands".to_string()));
        assert!(names.contains(&"ui_control".to_string()));
        assert!(names.contains(&"list_functions".to_string()));

        let legacy = openai_tools(false, false);
        let legacy_names = tool_schema_names(&legacy);
        assert_eq!(legacy_names.len(), 13);
        assert!(!legacy_names.contains(&"execute_commands".to_string()));
        assert!(!legacy_names.contains(&"ui_control".to_string()));
    }

    fn tool_schema_names(tools: &Value) -> Vec<String> {
        tools
            .as_array()
            .map(|rows| {
                rows.iter().filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str)).map(str::to_string).collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn catalog_prompt_groups_by_area_and_respects_the_budget() {
        let catalog = parse_catalog(&json!([
            {"id": "file.exportPdf", "label": "Export as PDF", "menu": ["File"], "params": ""},
            {"id": "home.bold", "label": "Bold", "menu": ["Home"], "params": ""},
            {"id": "data.sort", "label": "Sort…", "menu": ["Data"], "params": ""},
        ]));
        let prompt = catalog_prompt(&catalog, 4096);
        assert!(prompt.contains("[File] "));
        assert!(prompt.contains("[Home] "));
        assert!(prompt.contains("file.exportPdf"));
        assert!(prompt.contains("home.bold"));
        let tiny = catalog_prompt(&catalog, 30);
        assert!(tiny.contains("truncated"));
    }

    #[test]
    fn feature_guide_and_cheat_sheet_name_the_generic_tools() {
        for needle in ["execute_command", "execute_commands", "list_commands", "describe_command", "ui_control"] {
            assert!(FEATURE_GUIDE.contains(needle), "missing {needle}");
        }
        for needle in ["home.formatCells", "data.sortAscending", "insert.pivotTable", "file.exportPdf", "view.freezePanes"] {
            assert!(CHEAT_SHEET.contains(needle), "missing {needle}");
        }
    }

    #[test]
    fn guide_covers_the_upstream_features_added_after_the_fork_point() {
        // GridCraft 0.12/0.13 additions must be reachable through the same generic tools; the prompt
        // tells the model they exist so it does not report them as missing.
        for needle in [
            "stacked100",
            "headerFooter",
            "outlineSettings",
            "dateUnit",
            "enableShowDetails",
            "errorChecking",
            "checkForUpdates",
            "commandSearch",
            "interfaceScale",
        ] {
            assert!(FEATURE_GUIDE.contains(needle) || CHEAT_SHEET.contains(needle), "the AI guide does not mention the newer feature `{needle}`");
        }
    }

    #[test]
    fn tool_output_truncation_states_its_limit() {
        let text = "a".repeat(100);
        assert_eq!(truncate_tool_output(&text, 200), text);
        let cut = truncate_tool_output(&text, 10);
        assert!(cut.starts_with(&"a".repeat(10)));
        assert!(cut.contains("truncated"));
    }

    #[test]
    fn ground_rules_cover_intent_handling_and_report_generation() {
        for needle in [
            "Workbook Agent",
            "WORK ORDER",
            "QUESTION ABOUT THE WORKBOOK",
            "OUT-OF-TOPIC",
            "EMPTY WORKBOOK",
            "Always reply in English",
            "never misreport",
        ] {
            assert!(GROUND_RULES.contains(needle), "ground rules missing `{needle}`");
        }
    }

    #[test]
    fn layout_prompt_requires_the_final_pass_within_budget() {
        let total = GROUND_RULES.len() + FEATURE_GUIDE.len() + CHEAT_SHEET.len();
        assert!(total < 7000, "AI prompt grew heavy again: {total} chars");
        for needle in [
            "FINAL LAYOUT PASS",
            "ATURAN KERAPIAN MUTLAK",
            "create_report",
            "layout_sheet",
            "no fills",
            "blank 10pt spacer",
            "double",
            "indent",
            "no .00",
        ] {
            assert!(CHEAT_SHEET.contains(needle), "layout prompt missing `{needle}`");
        }
        assert!(FEATURE_GUIDE.contains("layout pass"), "feature guide must require the final layout pass");
    }

    #[test]
    fn report_columns_classify_for_alignment_and_formats() {
        assert_eq!(classify_report_column("Product", &[]), ReportColumnKind::Label);
        assert_eq!(classify_report_column("Category", &[]), ReportColumnKind::Text);
        assert_eq!(classify_report_column("Qty", &[]), ReportColumnKind::Quantity);
        assert_eq!(classify_report_column("Price", &[]), ReportColumnKind::Money);
        assert_eq!(classify_report_column("Margin (%)", &[]), ReportColumnKind::Percent);
        assert_eq!(classify_report_column("Code", &[]), ReportColumnKind::Code);
        // A numeric column without a telling header still formats/aligns as money.
        assert_eq!(classify_report_column("FY 2026", &[json!(450000000)]), ReportColumnKind::Money);
        // Header and data alignment must form one straight vertical line.
        assert_eq!(ReportColumnKind::Label.header_align(), "Left");
        assert_eq!(ReportColumnKind::Label.data_align(), "Left");
        assert_eq!(ReportColumnKind::Money.header_align(), "Right");
        assert_eq!(ReportColumnKind::Money.data_align(), "Right");
        assert_eq!(ReportColumnKind::Percent.data_align(), "Right");
        assert_eq!(ReportColumnKind::Code.header_align(), "Center");
        assert_eq!(ReportColumnKind::Code.data_align(), "Center");
    }

    #[test]
    fn formatted_widths_avoid_shebang_and_trailing_double_zero() {
        // Whole money/quantities are grouped but never show `.00`.
        assert_eq!(report_formatted_length(&json!(450000000), ReportColumnKind::Money, false), 12);
        assert!(!ReportColumnKind::Money.number_format(false).contains(".00"));
        assert!(!ReportColumnKind::Quantity.number_format(false).contains(".00"));
        assert!(!report_money_has_cents(&[json!(450000000), json!(-4500)]));
        assert!(report_money_has_cents(&[json!(12.5)]));

        // Width = formatted content/header + safe padding, never below the kind's floor.
        assert!(report_column_width("Total", &[json!(450000000)], ReportColumnKind::Money) >= 18);
        assert!(report_column_width("Product", &[json!("Gaming Laptop")], ReportColumnKind::Label) >= 38);
        assert!(report_column_width("Category", &[json!("Electronics")], ReportColumnKind::Text) >= 18);
        assert!(
            report_column_width("Long header name", &[json!(1)], ReportColumnKind::Quantity) >= "Long header name".len() as u32 + 6,
            "the header length must be included"
        );
        // Percent uses one decimal + % and stays right aligned.
        assert_eq!(report_formatted_length(&json!(0.253), ReportColumnKind::Percent, false), 6);
        assert_eq!(ReportColumnKind::Percent.header_align(), "Right");
    }

    #[test]
    fn row_roles_and_section_spacers_follow_the_layout_standard() {
        assert_eq!(column_letters(0), "A");
        assert_eq!(column_letters(25), "Z");
        assert_eq!(column_letters(26), "AA");
        assert_eq!(column_letters(27), "AB");

        assert!(report_is_total_label("TOTAL ASSETS"));
        assert!(report_is_total_label("Subtotal"));
        assert!(report_is_total_label("Net income"));
        assert!(report_is_total_label("Grand total"));
        assert!(report_is_total_label("OPERATING INCOME"));
        assert!(report_is_total_label("INCOME BEFORE TAX"));
        assert!(report_is_total_label("INCOME TAX"));
        assert!(!report_is_total_label("Revenue"));
        assert!(report_is_section_row(&[json!("REVENUE"), Value::Null, json!("")]));
        assert!(!report_is_section_row(&[json!("REVENUE"), json!(100)]));
        assert!(!report_is_section_row(&[json!("TOTAL"), Value::Null, Value::Null]));

        let labels: Vec<Option<String>> =
            ["REVENUE", "Product sales", "TOTAL REVENUE", "COST OF GOODS SOLD", "Raw materials", "TOTAL COGS", "GROSS PROFIT", "NET INCOME"]
                .iter()
                .map(|label| Some((*label).to_string()))
                .collect();
        let sections = vec![true, false, false, true, false, false, false, false];
        let roles = report_row_roles(&labels, &sections);
        assert_eq!(roles[0], ReportRowRole::Section);
        assert_eq!(roles[1], ReportRowRole::LineItem);
        assert_eq!(roles[2], ReportRowRole::Subtotal);
        assert_eq!(roles[3], ReportRowRole::Section);
        assert_eq!(roles[6], ReportRowRole::Subtotal);
        assert_eq!(roles[7], ReportRowRole::GrandTotal);

        let spacers = report_spacer_after(&roles, &sections);
        assert!(spacers[2], "one spacer after a section subtotal before the next section");
        assert!(!spacers[5], "no free spacer before GROSS PROFIT, only section boundaries");
    }
}
