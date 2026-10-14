# Design

Loom: https://www.loom.com/share/040de97ef5cb4548b1969c1584c44614

## Summary (High-Level Flow)

1. POST /chat receives a new call or message.
2. The message goes to a router LLM, which sees a compact index of all 191 tools.
3. The router LLM can choose up to 15 tools as its output.
4. The agent loop's context is the past messages with compaction + up to 15 tools given + 1 load_more_tools (if it needs more tools).
5. The agent can choose between two actions:
   - Call a tool (sequentially)
   - Text reply (ambiguity or finish)

## Tool Routing

**Problem:** There are 191 tools, which cost tokens on every call and reduce the effectiveness of the agent.

**Options considered:**

1. Service-level routing
2. Per-service sub-agents
3. LLM selector over compacted schemas

**What is chosen:** Option 3.

- One router LLM call reads the messages and the name + one-line description of all 191 tools, and returns up to 15 tools.
- load_more_tools is added to let the agent get more tools as needed.
- Output is constrained to the enum of registered names.
- Failing case:
  - The router gets two attempts, each limited by the 60-second turn budget.
  - If both fail, a keyword-based fallback runs.
  - If no keyword matches, it sends back an empty list.

**Why:**

- It handles multi-step prompts and is cheaper in tokens and time.
- Service-level routing still needs internal routing, because GitHub has 80+ tools.
- Sub-agents make simple tasks more complex and cost more tokens and time.

**What is still breaking:**

- The router adds an extra layer to the call on every prompt.
- The router chooses before any result is seen, so there is a cost for loading more tools if the tools given are not enough.
- Tools are never dropped, so any wrong pick is kept and the list only grows over the turn.
- No routing metric or logging yet. Testing was only run on the end-to-end live scenarios.

## Orchestration Loop

**Shape:**

- A single ReAct agent (model call → tool calls → results appended → repeat)
- Writes run directly on an explicit request, and every call goes through one wrapper.

**Stops when:**

- The model replies with no tool calls, either an answer or a question
- The 12-turn or 60-second limit is reached
- An identical call is refused after two runs

**Errors:**

- Structured errors, where different raw errors are routed to different actions on what to do.
- Two kinds of failure in the log:
  - A tool that ran and raised is logged with the error and a null result.
  - A call that code refused before it ran is not logged, because nothing was invoked.

**Ambiguity:**

- Prompt rules
- Code will refuse a write before it runs if:
  - An ID, recipient or similar reference did not come from the conversation or a tool result
  - A content argument is blank
  - An end time is not after its start time
- Tool results are wrapped as data to avoid prompt injection.

**Parallel vs Sequential:**

- Code runs a batch sequentially to keep the order deterministic.

**What is still breaking:**

- No completeness check on a normal stop: nothing verifies that every part of a multi-part request was done.
- Failed-write check is by tool name: if a post to one channel fails and a post to another succeeds, the first failure is cleared from the prefix.
- A batch is not cut short: if the first call in a batch fails, the rest still run. A dependent write in the same batch is stopped only by the grounding gate.
- Long histories: tool results are trimmed for the model, but history summarization is not built yet.

## Tradeoffs Discarded

- Confirming before writes: the grader is single-turn, so a confirmation turn would fail the scenario.
- A session store: the request already carries the full history.
- Inferring read/write from verb names: it misclassified about a dozen tools, so the table of 107 read, 69 write and 15 destructive is explicit, and unknown names count as destructive.

## With Another 8 Hours

- History summarization, which a long-running agent would need.
- gpt-4.1-mini still ends some replies with "Would you like me to…?" instead of just ending.
- Evals on different router and agent models, to cut cost and reduce test errors.
- Parallel execution for independent reads, and approval steps for writes and deletes in production.
