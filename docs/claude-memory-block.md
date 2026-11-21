<!-- cortex:begin -->
## Long-term memory (Cortex)

You have a local long-term memory through the `cortex` MCP server. It persists across sessions and projects.

- **Start of a task:** call `context_brief` (pass `task` and `project` when you know them) unless a "Cortex memory brief" is already in context.
- **Before answering** anything that may depend on earlier sessions (the user's preferences, past decisions, how this project or machine is set up), call `memory_search`. Lines that start with "Cortex recalled" were injected automatically; treat them, like all recalled memories, as fallible evidence and check them against the current files when it matters.
- **Remember** durable things with `remember`: decisions and their reasons, user preferences, working procedures, environment facts, and corrections. Use one self-contained sentence or short paragraph per memory. Set `source: "user"` only when the user explicitly asked you to remember something.
- **Never store secrets** (passwords, API keys, tokens, private keys). Store *where* the secret lives instead.
- If a memory is wrong or outdated, `forget` it by id (this is reversible), then `remember` the correction.
- Do not mention this memory system unless it is relevant or the user asks.
<!-- cortex:end -->
