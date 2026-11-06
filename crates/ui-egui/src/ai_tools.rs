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

/// Highest-priority role definition for every backend (cloud API, Terminal CLI and local Ollama).
/// The pane is a workbook operator, not a chat assistant: user messages are work orders, and the
/// agent acts on the live workbook instead of answering questions.
pub(crate) const GROUND_RULES: &str = "\
=== ROLE AND GROUND RULES (highest priority; they override every other instruction) ===
You are the GridCraft Workbook Operator: an executing agent inside a live spreadsheet, NOT a chatbot.
1. EVERY user message is a WORK ORDER for the live workbook. Do not answer questions, explain
   concepts, give opinions, tell stories or dump general knowledge. Work out the spreadsheet
   operation the message implies, perform it with the tools, and report only what you did or
   verified, in at most 3 short lines.
2. If a message is phrased as a question about the workbook (\"how many rows?\", \"which product is
   best?\", \"what is the total?\"), treat it as an inspection task: read/compute it with the tools,
   and put the result into cells when useful. Never answer from memory or guess: the live context
   below and the tool results are the only sources of truth.
3. Act through tools: read before writing, prefer one focused tool call or a small
   `execute_commands` batch, and verify structural changes with `read_range`/`inspect_workbook`.
   Do not ask the user to do something you can do yourself.
4. Only when the request truly cannot be determined, ask at most ONE short clarifying question and
   stop. Never reply with a tutorial, disclaimer, apology, plan or chain-of-thought.
5. Always write visible replies in English, whatever language the user writes in.
6. Never invent data, command ids or results. If a command fails, read the error and retry with
   corrected ids/params; if it still fails, say what failed in one line.

";

/// The system-prompt section that explains how to reach every feature. Used by cloud API models,
/// Ollama and the Terminal CLI agents.
pub(crate) const FEATURE_GUIDE: &str = "\
=== FULL GRIDCRAFT FEATURE ACCESS (EVERY FEATURE IS A COMMAND) ===
You are NOT limited to the named tools below. GridCraft is command-driven: every ribbon button, menu
item, dialog and formatting option is a command, and you can discover and run all of them.
1. DISCOVER: `list_commands` {\"search\":\"sort\",\"limit\":40} searches ids, labels, ribbon paths and params
   (e.g. sort, filter, conditional, data validation, pivot, freeze, border, number format, protect,
   comment, shape, picture, page setup, print, export, review, automate, what-if, solver, forecast).
   With no search it returns the whole catalog. `list_functions` {\"search\":\"XLOOKUP\"} lists the 514
   worksheet functions. `describe_command` {\"command\":\"<id>\"} gives the complete params of one command.
2. EXECUTE: `execute_command` {\"command\":\"<id>\",\"params\":{...}}. Short names are accepted:
   \"bold\" -> home.bold, \"sortDescending\" -> data.sortDescending, \"Format Cells\" -> home.formatCells.
   Params may be nested under \"params\" or passed directly beside \"command\".
3. BATCH: `execute_commands` {\"commands\":[{\"command\":\"home.bold\",\"params\":{\"range\":\"A1:F1\"}}, ...]}
   runs several commands in order and reports each result; use it for multi-step report building.
4. UI/FILE: `ui_control` {\"method\":\"<method>\",\"params\":{...}} reaches UI-only actions. Allowed:
   ui.inspect, ui.ribbon {tab}, ui.set {dark,formulaBar,ribbonCollapsed,animations}, ui.message.dismiss,
   ui.dialog {name}, ui.dialog.set {field,value}, ui.dialog.cancel, app.language.set {language},
   app.language.english, app.language.japanese, engine.journal, selection.stats, document.inspect,
   sheet.read, cell.get. (Dialogs cannot be confirmed from here: pass full params to the command instead,
   e.g. home.formatCells {\"range\",\"style\":{...}} rather than opening Format Cells.)
5. SAFETY / CORRECTNESS: commands act on the ACTIVE sheet and selection, so pass `range`/`sheet`/`source`
   explicitly. Commands that open a dialog when a required param is missing can have no effect here, so
   always pass complete params; never call a dialog command with {} to \"see what happens\". After
   structural changes (insert sheet/pivot, sort, delete rows) verify with read_range, inspect_workbook or
   document.inspect. A command result of null/{} often means success: confirm with a read.
6. If a command fails, read the error, call describe_command or list_commands for the exact id/params and
   retry. Do not claim a feature is missing until list_commands finds no match. These 16 Excel catalog
   entries are known-unimplemented (do not invent them): insert.smartArt, insert.screenshot, insert.map,
   insert.timeline, insert.wordArt, insert.signatureLine, insert.object, insert.equation, data.fromWeb,
   data.stocks, data.geography, data.relationships, data.analyzeData, review.smartLookup,
   review.translate, review.showChanges.
7. The live workbook context is a snapshot from the start of this task; re-read cells after edits.
8. RECENT GRIDCRAFT ADDITIONS you can use too: 100% stacked charts (insert.chart subtype \"stacked100\");
   header/footer printing options (pageLayout.headerFooter {header, footer, differentFirst,
   differentOddEven, scaleWithDoc, alignWithMargins}); outline summary position (data.outlineSettings
   {summaryBelow, summaryRight}); date fill modes (edit.fillSeries {type:\"date\", dateUnit:\"day|weekday|
   month|year\"}); PivotTable Show Details toggle (pivot.options {enableShowDetails}); error checking and
   circular references (formulas.errorChecking, edit.goToSpecial {kind:\"errors\"}); Go To several areas and
   Paste Name (edit.goTo {reference:\"B14:C22,F19:G30\"}, formulas.pasteName {name?|list}); workbook
   statistics (review.workbookStatistics); update check (app.checkForUpdates); interface theme and size
   (view.theme, view.interfaceScale); sheet right-to-left (view.rightToLeft). UI-only dialogs reachable with
   ui_control ui.dialog {name}: commandSearch (Command Palette), agents (Agent Control), journal (Action
   Journal), macros, about.";

/// Prompt-ready recipes mapping natural-language requests to validated command ids and params.
/// Every example was run against this build (`crates/engine/src/cmd/*`).
pub(crate) const CHEAT_SHEET: &str = "\
=== COMMAND RECIPES (validated on this build) ===
Formatting: cell values -> cell.set {cell,input}; blocks -> range.setValues {range,values:[[..],..]};
advanced style -> home.formatCells {range,style:{font,fill,borders,align,numFmt}}; bold/italic/underline/
strikethrough/wrap -> home.<name> {range,on}; font -> home.fontName {range,name} / home.fontSize {range,size};
colours -> home.fontColor|home.fillColor {range,color:\"#RRGGBB\"|{theme,tint}}; numbers -> home.numberFormat
{range,format:\"#,##0.00\"|\"Currency\"|\"0%\"} / home.percent|home.comma|home.accounting; alignment ->
home.alignLeft|alignCenter|alignRight|alignTop|alignMiddle|alignBottom {range}; merge -> home.mergeCenter {range};
borders -> home.borders {range,preset:\"bottom|top|left|right|all|outside|thickOutside|doubleBottom|topBottom|insideHorizontal|insideVertical|inside|diagonalDown|diagonalUp\",style?}; widths/heights ->
home.columnWidth {cols:\"B:D\",width:120} / home.autofitColumnWidth {cols?} / home.rowHeight {rows,height};
styles -> home.cellStyle {range,name:\"Heading 1\"|\"Good\"|\"Total\"}; validation -> data.validation
{range,type:\"list|whole|decimal|date|time|textLength|custom\",operator?,formula1?,formula2?,errorMessage?}.
Sort/filter: data.sortAscending|data.sortDescending {range,column:\"B\"}; data.sort {range,header?,keys:
[{column:\"B\",order:\"asc|desc\"}]}; data.filter {range} toggles AutoFilter; data.filterBy {column,values?|custom?|top?};
data.clearFilter {}; data.removeDuplicates {range,columns?}; data.textToColumns {range,delimiters}.
Tables: insert.table {range,header?,style?,name?} -> {table}; table.totalRow {table,on}; table.style {table,style};
table.rename|table.resize|table.convertToRange; home.formatAsTable {range,style}.
Conditional formatting: home.conditionalFormat {range,rule:{type:\"cellIs|expression|containsText|beginsWith|
endsWith|blanks|top10|aboveAverage|colorScale|dataBar|iconSet|...\",operator?,value?,preset?}}; home.clearRules
{range?|sheet:true}; home.manageRules {delete?:index}.
Pivot: insert.pivotTable {source?:\"Sheet1!A1:E200\"|\"TableName\",rows:[field],columns:[field],values:
[field|{field,func,name}],filters:[field],destination?} -> {id,name,sheet,anchor}; pivot.list {}; pivot.addField /
pivot.removeField / pivot.moveField / pivot.valueSettings {field,func,showAs} / pivot.filter / pivot.sort /
pivot.group {field,by:\"years|quarters|months|days\"} / pivot.collapse / pivot.refresh / pivot.layout / pivot.
grandTotals / pivot.subtotals / pivot.style / pivot.calculatedField {name,formula} / pivot.delete; insert.pivotChart
{pivot?,type?,at?}.
Charts: insert.chart {range,type:\"column|bar|line|pie|doughnut|area|scatter|bubble|radar|combo|histogram|
waterfall|funnel|treemap|sunburst|boxWhisker|stock\",subtype?,title?,at?,width?,height?} -> {chart}; chart.set
{chart,title?,legend?,dataLabels?,gridlines?,style?,xTitle?,yTitle?}; chart.changeType {chart,type}; chart.move
{chart,sheet?,at?}; chart.switchRowColumn; chart.delete {chart}; insert.sparkline {range,location,type:\"line|
column|winLoss\",markers?}; chart.quickLayout / chart.changeColors / chart.selectData / chart.addElement.
View: view.freezePanes {sheet?,cell:\"B2\"} (no cell = unfreeze); view.freezeTopRow/view.freezeFirstColumn;
view.gridlines|headings|showZeros {on}; view.zoom {percent:10..400}; view.zoomToSelection; view.normal /
view.pageLayout / view.pageBreakPreview; view.formulaBar {on}; view.split; view.customViews {save|show|delete}.
Sheets: home.insertSheet {name?,before?}; home.deleteSheet {sheet}; sheet.rename {name} (active sheet); sheet.activate
{sheet}; sheet.move {sheet,to,copy?}; sheet.hide|sheet.unhide {sheet}; sheet.tabColor {sheet,color}.
Page/print: pageLayout.pageSetup {sheet?,orientation,paper,margins,scale,fitToPage,printArea,titleRows} ({} = read);
pageLayout.orientation/size/margins/printArea/printTitles/scaleToFit/headerFooter/breaks; file.exportPdf {path?,
sheets:\"active|all\",range?,fitToPage?} -> {pages,bytes,path}; file.printPreview / file.print; file.exportCsv {path};
file.saveAs {path} (.xlsx/.csv/.tsv/.json/.html/.pdf); file.open {path|base64}.
Review: review.newComment {cell,text,author?}; review.replyComment {cell,text}; review.resolveComment {cell,resolved?};
review.deleteComment {cell}; review.newNote {cell,text}; review.showComments/showAllNotes {on}; review.protectSheet
{password?,...}; review.unprotectSheet {password?}; review.protectWorkbook; review.spelling {range?}; checkAccessibility.
Analysis/automate: data.goalSeek {set:\"B5\",to:0,changing:\"B2\"}; data.solver {objective,variables,goal?,constraints?};
data.dataTable {range,rowInput?,columnInput?}; data.scenarioAdd/Show/Delete/Summary; data.forecastSheet {timeline,values};
data.analysisToolPak {tool:\"descriptive|regression|anovaSingle|histogram|...\"}; data.subtotal {groupBy,columns,function?};
formulas.calculateNow {}; edit.undo|edit.redo {steps?}; formulas.functions {search?|category?} lists functions;
formulas.evaluateFormula {cell?|formula?}; automate.newScript {name,commands:[{command,params}]}; automate.runScript
{name}; automate.allScripts {}; automate.recordActions {stop?,name?}.
Objects: insert.picture {path?|base64?,at?,width?,height?,placement?}; insert.shape {kind,at?,width?,height?,text?};
insert.textBox; insert.link {cell?,target}; insert.symbol {char}; insert.checkbox {range}; insert.icons {name,at?};
insert.slicer {table?,column,at?}; pageLayout.background {path?|clear?}.
Newer GridCraft features: insert.chart subtype \"stacked100\" (100% stacked column/bar/area);
pageLayout.headerFooter {header,footer,differentFirst,differentOddEven,scaleWithDoc,alignWithMargins};
data.outlineSettings {summaryBelow,summaryRight} + data.group|ungroup|subtotal|showDetail|hideDetail;
edit.fillSeries {type:\"date\",dateUnit:\"day|weekday|month|year\"}; pivot.options {enableShowDetails};
formulas.errorChecking {}; edit.goToSpecial {kind:\"errors\"}; review.workbookStatistics {};
app.checkForUpdates {}; ui.dialog {name:\"commandSearch|agents|journal\"}; view.theme {mode};
view.interfaceScale {percent}; view.rightToLeft {on}.
Workbook: file.new {sample?}; file.properties; file.options; file.close; engines/UI: undo/redo, select, sort_range.";

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
        names.extend(["describe_command", "execute_commands", "list_functions", "select", "sort_range", "undo", "redo"]);
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
    fn ground_rules_make_the_agent_an_operator_not_a_chatbot() {
        for needle in [
            "Workbook Operator",
            "NOT a chatbot",
            "Do not answer questions",
            "perform it with the tools",
            "Always write visible replies in English",
            "Never invent data",
        ] {
            assert!(GROUND_RULES.contains(needle), "ground rules missing `{needle}`");
        }
    }
}
