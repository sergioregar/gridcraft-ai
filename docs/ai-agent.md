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

Every backend gets the same **ground rules** at the top of its prompt:

- The agent is the **GridCraft Workbook Operator**, not a chatbot. Every user message is a work
  order for the live workbook: it performs the spreadsheet operation with tools and reports what it
  did — it does not answer general questions, explain concepts or dump knowledge.
- Messages phrased as workbook questions ("how many rows?", "which product is best?") are treated
  as inspection tasks: read/compute them with the tools (and put the result in cells when useful),
  never answer from memory.
- The agent reads before writing, prefers one focused call or a small `execute_commands` batch,
  verifies structural changes, and asks at most one short clarifying question when the request
  cannot be determined.
- Visible replies are always in **English**, whatever language the user writes in, and are kept to
  a few short lines.

Local Ollama models additionally receive a note that the context and command catalog in the prompt
are complete for the active sheet, and to take one step at a time — the weaker models need that
explicit framing.

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
