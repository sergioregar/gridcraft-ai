# AI Agent

GridCraft's built-in AI Agent drives the live workbook from the **AI Agent** pane (Automate › AI
Agent). It has three backends, and each is configured in its own tab:

| Tab | Backend | How it connects |
|---|---|---|
| LLM API Key | OpenRouter, AgentRouter, OpenAI, Anthropic, Gemini, DeepSeek, Groq, or a custom OpenAI-compatible endpoint | Enter an API key, pick a model (dropdown or typed by hand), then Connect. |
| Terminal Agent | `agy`, `claude`, `codex`, or a custom CLI | The agent runs the CLI once per turn and reads `<gridcraft_tool>` tags from its reply. |
| Local (Ollama) | A local Ollama server (`http://127.0.0.1:11434` by default) | Uses Ollama's OpenAI-compatible endpoint, with the same tool set as the API tab. |

Switching tabs never disconnects a running backend. The pane stays deliberately short for
non-technical users: the backend tabs and their connection form, the connection status, and one
**History** subsection that lists what the agent was asked and what it did. There is no separate
chat/MCP panel and no access-control card to configure.

## Role and ground rules

Every backend gets the same compact **ground rules** at the top of its prompt; the agent decides the
intent of each message first:

- **Work order** (create, edit, format, sort, filter, delete, chart, pivot, export…) → perform it with
  tools on the live workbook and report what it did in a few short lines.
- **Question about the workbook** ("how many rows?", "best seller?", "total?") → inspect or compute it
  with tools and answer briefly from the tool results (writing the result into cells when useful).
- **Question about GridCraft or spreadsheets** ("how do I freeze panes?") → a 1–3 line answer plus an
  offer to do it; no tutorials.
- **Out-of-topic question** (news, general knowledge, coding help, chit-chat) → declined in one line
  ("I only work on this workbook.") and nothing is run.
- **Empty workbook + a request to make a report/list/data** → the agent creates the data itself: a
  title, a header row, at least ten plausible rows (unique products/categories, quantities, unit
  prices, dates), formulas, a totals row, number formats, a table style and a chart when asked.
  "Never invent data" only forbids misreporting existing data, not generating what the user asked for.

The agent reads before writing, prefers one focused call or a small `execute_commands` batch, verifies
structural changes, and asks at most one short clarifying question when the intent cannot be
determined. Visible replies are always in **English**, whatever language the user writes in.

The prompt is deliberately light: the ground rules and the system prompt are compact, the command
catalog is capped (12,000 characters for cloud/terminal agents, 5,000 for local Ollama), and local
models get the catalog plus a short "take one step at a time" note instead of the full guide and
recipe sheet. This keeps small local models focused on the task instead of drowning in prompt text.

## Creating data from an empty sheet

The engine and the agent both support starting from scratch. A validated recipe the prompt teaches:

1. `write_range` at `A1` with a title row, a header row and 10+ products, e.g. `[["Sales Report"],
   ["Product","Category","Qty","Price","Total"], ["Gaming Laptop","Electronics",12,18500000,"=C3*D3"], …,
   ["TOTAL","","","","=SUM(E3:E12)"]]`.
2. `home.bold`/`home.fillColor`/`home.fontColor` on the title and header, `home.numberFormat`
   (`#,##0.00`) on the money columns, `home.borders` on the block, `insert.table` for a table style.
3. `insert.chart` (`{"range":"B3:E12","type":"column","title":"Sales by Product","at":"G2"}`) when a
   chart is wanted, then `file.saveAs` / `file.exportPdf`.

The same sequence is reproducible without the AI: `gridcraft-cli run --script` with `range.setValues`,
`home.*`, `insert.table` and `insert.chart` builds the identical report.

## Choosing a model (LLM API tab)

The **Model** dropdown starts with the provider's built-in candidates and is replaced by the exact
list the key can use after the first `GET {base_url}/models` call (the **Refresh** button reloads it
at any time; Connect also fills it). The **Custom model** field accepts any id the provider does not
list (a preview model, a self-hosted name, an alias). The chosen id is sent as `model` on every chat
request, and `Connect` verifies that model first, then falls back to the account's other models and
the built-in candidates.

DeepSeek, for example, lists `deepseek-chat` and `deepseek-reasoner`; the dropdown shows whatever
the key actually returns, and typing `deepseek-reasoner` works even before a list is loaded.

```json
{"command": "ui.aiAgent", "params": {"provider": "DeepSeek", "model": "deepseek-chat"}}
```
Agents can preselect the provider and model over the control channel (the API key is never a
command parameter and never reaches the journal).

## Full feature access (always on)

Every GridCraft feature is a command (see [`architecture.md`](architecture.md) and
[`control-protocol.md`](control-protocol.md)), so the agent is not restricted to a fixed set of
actions. Full access is on by default and does not need a switch; the agent can:

- discover the whole surface — `list_commands` searches all engine and UI commands by id, label,
  ribbon path and params; `describe_command` returns one command's full documentation;
  `list_functions` searches the 514 worksheet functions;
- run any command — `execute_command` (single) and `execute_commands` (batch, with per-command
  results). Short names work: `bold` → `home.bold`, `sortDescending` → `data.sortDescending`,
  `Format Cells` → `home.formatCells`. Unknown ids come back with the nearest candidates instead of
  a bare "unknown command";
- reach UI-only actions — `ui_control` covers ribbon/view state, interface language, read-only
  workbook queries and opening dialogs for inspection. Pointer/keyboard injection, printing,
  quitting the app and a dialog's OK button stay out of reach on purpose: commands express those
  intents with explicit parameters, where the agent has to say what it wants;
- build multi-step work — `select`, `sort_range`, `undo`/`redo`, and the chart/table helpers mirror
  the MCP server's tools ([`mcp.md`](mcp.md)); anything else is one `execute_command` away.

The live command catalog is embedded in the prompt so a model knows a feature exists without
spending a tool round trip. Terminal CLI agents get the same catalog and recipes inline, because
they cannot call back into GridCraft mid-turn; if every tool in an attempt fails they get one more
attempt with the errors in the prompt. New commands that land in later GridCraft versions are
discovered at runtime, so they are usable by the agent without changing the pane.

## What the agent can rely on

- The live grid/table context (sheets, tables, column roles, cell values) is captured when a turn
  starts; the agent is told to re-read after structural edits.
- Commands act on the active sheet and selection, so the prompt requires explicit `range`/`sheet`
  parameters.
- Commands that open a dialog when a required parameter is missing have no effect in the agent's
  headless control path; the prompt requires complete parameters and the agent can call the
  command's primitive form instead (for example `home.formatCells {range, style}` rather than
  opening Format Cells).
- Because full access is on, the agent runs exactly the command it asked for: the legacy auto-wipe
  and chart-substitution heuristics are off. A malformed `ai.deep_clear` range is reported as an
  error instead of clearing the whole sheet.

## Privacy

The pane only talks to the backend the user configures. There is no telemetry; local Ollama keeps
the prompt on the machine. The API key lives in the session's memory (`AiAgentState`), not in
`ui.json`/`prefs.json`.
