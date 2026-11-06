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

The prompt is deliberately light: the ground rules and the system prompt are compact, the command
catalog is capped (12,000 characters for cloud/terminal agents, 5,000 for local Ollama), and local
models get the catalog plus a short "take one step at a time" note instead of the full guide and
recipe sheet. This keeps small local models focused on the task instead of drowning in prompt text.

## Per-file sessions

Every workbook document has its **own AI chat session**: History and the input draft are keyed by the
document, so creating a new file or opening another file starts with an empty History, and switching
back to an earlier file restores that file's conversation. The live workbook context (sheets, cells,
formulas, selection) is rebuilt from the active document on every turn anyway, so the agent never
carries another file's data as context.

Switching files while a turn is running stops that turn and parks its history with the file it
started on (the file gets a short "Task stopped because the active file changed." note); the new
file's session never receives another file's messages. Closing a document forgets its session.

Backend configuration — provider, API key, CLI agent/model, Ollama URL and connection status — stays
global and is configured once, not per file. Only the conversation/context is per file; if you need
per-file backends as well, that would be a separate change.

## Voice input

A microphone button sits between the chat input and **Send**, and a **Mic** selector below the box
chooses which recording device to use (with **Refresh** to re-scan the devices). Press the mic, speak
one utterance, then press it again; the local speech model transcribes the recording and writes the
text into the input box. **Nothing is sent automatically** — review or edit the text and press
**Send**/Enter yourself. A second voice recording appends to whatever is already typed.

Transcription uses the local, lightweight **multilingual Whisper** model (`tiny`, run through
`faster-whisper` on the CPU): it works offline, detects the spoken language automatically (English,
Indonesian and many others) and audio never leaves the machine. The first use downloads the model
(~75 MB) once; a recording is transcribed in about a second on a laptop CPU. Set
`GRIDCRAFT_WHISPER_MODEL` (e.g. `base` or `small`) for a larger, more accurate model.

Requirements on the desktop: Python on `PATH` with `faster-whisper` installed
(`pip install faster-whisper`). If Python or the package is missing, the pane shows a one-line error
with the exact command; microphone failures (unplugged device, wrong input) are reported there too.
Voice input is desktop-only (not in the web build).

## Creating data from an empty sheet

## Creating data from an empty sheet

The engine and the agent both support starting from scratch. A validated recipe the prompt teaches:

1. `write_range` at `A1` with a title row, a header row and 10+ products, e.g. `[["Sales Report"],
   ["Product","Category","Qty","Price","Total"], ["Gaming Laptop","Electronics",12,18500000,"=C3*D3"], …,
   ["TOTAL","","","","=SUM(E3:E12)"]]`.
2. `home.bold`/`home.fillColor`/`home.fontColor` on the title and header, `home.numberFormat`
   (`#,##0.00`) on the money columns, `home.borders` on the block, `home.autofitColumnWidth` for
   readable columns, `insert.table` for a table style.
3. `insert.chart` (`{"range":"B3:E12","type":"column","title":"Sales by Product","at":"G2"}`) when a
   chart is wanted, then `file.saveAs` / `file.exportPdf`.

The same sequence is reproducible without the AI: `gridcraft-cli run --script` with `range.setValues`,
`home.*`, `insert.table` and `insert.chart` builds the identical report.

## Report layout tools (final layout pass)

Hand-placing cells one by one is where models produce cramped columns, `###` and clipped labels, so
the agent has two layout-first tools. Both run as the **final pass**, after every value and formula is
written, and both are **neutral**: no fills, font colours, table styles or theme are forced.

- **`create_report`** builds the structure in one call and then lays it out: merged left-aligned title
  (bold 15pt) and subtitle (italic 11pt) with no border, exactly one blank 10pt spacer row before the
  header row; a bold header row with top thin + bottom medium borders; section headings bold left at
  indent 0; line items regular left at indent 1; section subtotals bold with top + bottom thin
  borders; the bottom-most total bold with top thin + bottom **double** border; one blank 10pt
  spacer row between a closed section and the next section heading; number formats on the numeric
  columns; sizes/widths from the formatted display length plus padding; frozen header; print fit to
  one page wide.
- **`layout_sheet`** applies the same final pass to an existing range **without changing any value or
  formula**: it merges/styles the title block, inserts the missing 10pt spacer rows below a subtitle
  and between sections (`home.insertRows`, which shifts formula references correctly), sets row
  heights, alignment/indent, number formats, functional borders, frozen header and print setup.

Widths use the formatted display length, not the raw value: grouped thousands, signs/parentheses and
`%` are counted, and `+6` characters of padding are added so values never touch the next column.
Floors are 36–42 characters for label/description columns, 16–20 for numbers, 12–14 for
quantities/percentages. Whole money/quantities are formatted with `#,##0_);(#,##0);"-"` (never `.00`),
percentages with `+0.0%;-0.0%;0.0%`, and both are right aligned with the header so the header and its
column form one straight vertical line. Rows are vertically centred throughout; short headers are
never wrapped (the column is widened instead). There are no vertical borders and no per-cell grid
box: the only borders are the header rule, the subtotal rules and the grand-total double rule.

A prompt like "buatkan laporan penjualan 10 produk" should call `create_report` with the headers and
rows (or generate the data first) instead of writing cells individually; an income statement with
section headings is just rows where only the first cell has text. If a sheet already looks cramped,
`layout_sheet` fixes widths, heights, alignment and borders in place.


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

## Timeouts

Chat generations have **no request timeout**: the HTTP call to the provider (or local Ollama) and the
Terminal CLI turn wait as long as the model needs. Cancel/Stop still aborts them immediately, and
genuine LLM errors (HTTP status, auth/quota/model errors, network loss, invalid JSON) are still
reported. Only the short health probes keep a timeout: loading a model list and verifying a model
connection give up after a few seconds so a dead endpoint does not block the pane.
## Privacy

The pane only talks to the backend the user configures. There is no telemetry; local Ollama keeps
the prompt on the machine. The API key lives in the session's memory (`AiAgentState`), not in
`ui.json`/`prefs.json`.
