# Retail Analyst Agent — High-Level Design

A chat agent that lets non-technical retail executives analyse sales, customers and products in
BigQuery (`thelook_ecommerce`), discuss the results and get reports with action items.
This document is the production design; the [prototype](../README.md) implements a subset of it
(safety and PII, confirmed deletions, resilience, observability) with the same architecture.

**Contents**

1. [Summary](#1-summary)
2. [Architecture](#2-architecture)
3. [Technology choices](#3-technology-choices)
4. [Data flow](#4-data-flow)
5. [Requirements](#5-requirements)
6. [Error handling and fallbacks](#6-error-handling-and-fallbacks)
7. [Extensibility](#7-extensibility)
8. [Production operations](#8-production-operations)
9. [Setup and example run](#9-setup-and-example-run)
10. [Limitations and next steps](#10-limitations-and-next-steps)

---

## 1. Summary

**One analyst agent, every guarantee in code.** A single Gemini model plans the analysis, writes
SQL, reads the results and writes the answer, in a loop of tool calls (ReAct) inside a small
LangGraph graph. It is never trusted with anything that must not go wrong. The principle used
throughout is **"the model proposes, the code guarantees"**:

| Must never happen                   | Enforced by code, not by the prompt                                                                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A user sees another brand's data    | Every query is parsed and rewritten so each table contains only the user's brands                                                                        |
| PII appears in an answer            | Scoped tables have no PII columns; customer ids are salted pseudonyms; demographic groups below 5 customers are suppressed; output regex as a safety net |
| The database is changed             | Read-only credentials; only a single `SELECT` over four allowed tables passes the SQL guard                                                            |
| Reports are deleted without consent | Deletion pauses the graph; only the user's exact confirmation of the previewed list deletes                                                              |
| Costs explode                       | Dry-run cost check, a hard byte cap per query, a per-question limit of model calls, retries only for transient errors                                    |
| A failure is invisible              | Every step of every answer is traced with a trace id                                                                                                     |

The model is free where freedom helps: choosing what to query, exploring hypotheses for "why"
questions, and writing the narrative. It answers through a structured `submit_answer` tool that
separates **facts** (each citing the query that produced it) from **hypotheses**,
**recommendations** and **assumptions**, so the reader can always tell what the data shows from
what the model thinks.

**Prototype vs production.** It is one design. The prototype is a CLI over the same graph, tools
and guards; production adds the interface adapters, managed services and the quality layers that
need data or budget the prototype does not have (section 2.3).

---

## 2. Architecture

### 2.1 Production architecture

The system has two parts: the **request path** that answers a question in seconds, and
**background processes** that keep the knowledge, the persona and the quality checks up to date.

![Request path: how one question is answered](request-path.svg)

**How a question flows** (numbers match the diagram).

1. The web client sends the message with the user's JWT (issued by the client's frontend; its
   claims carry the user's role and brands). API Gateway validates the token against the
   identity provider's public keys (`chat-api` can verify it itself the same way); `chat-api`
   takes the user's identity and scope from the claims, never from the conversation. Progress is
   streamed over Server-Sent Events. Slack does not send a JWT: a separate **slack-adapter**
   verifies Slack's request signature, maps the Slack user to the user in the client's identity
   provider, acknowledges within Slack's 3-second limit, queues the turn in Cloud Tasks and edits
   its message as progress arrives (a Cloud Tasks delivery is an HTTP request, so Cloud Run keeps
   CPU allocated for the whole turn). In Slack, deletion is confirmed with buttons.
2. The agent calls Gemini on Vertex AI with the prompt and the tools; Gemini answers with tool
   calls.
3. Every SQL statement the model writes goes to the SQL guard.
4. Only the checked and rewritten statement reaches BigQuery, under a read-only service account.
5. Conversation state, saved reports, the deletion audit, preferences and persona versions are in
   Postgres, so any Cloud Run instance can continue any conversation, including one paused for a
   deletion confirmation.
6. Every step is traced.

![Background processes: keeping knowledge and quality up to date](background-processes.svg)

**Background processes.** Analysts approve new Question → SQL → Report trios into Cloud Storage;
an ingestion job validates each one with the same SQL guard the agent uses, dry-runs it, strips
numerical findings from the report, embeds it and publishes a new index file to Cloud Storage,
which `chat-api` loads into memory. The CEO's tone edits go through an automated quality gate and are published
as a new persona version without a redeploy. A nightly evaluation runs the golden test set.
Traces are exported to BigQuery; failed or corrected answers become candidate trios for the
analysts, which closes the learning loop (requirements 1 and 4).

**Where data lives.**

| Data                                                | Store                                   | Notes                                                                                         |
| --------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------- |
| Transactions                                        | BigQuery (`thelook_ecommerce`)        | read-only; authorized views without PII columns; aggregation threshold rule on the demographic view |
| Conversations, checkpoints                          | Cloud SQL Postgres                      | LangGraph Postgres checkpointer; deleted after 30 days                                        |
| Saved reports, audit log                            | Cloud SQL Postgres                      | reports private to the author; hard delete (see below); audit keeps who / when / which ids, never content |
| User preferences                                    | Cloud SQL Postgres                      | fixed fields + free-text notes (requirement 4)                                                |
| Persona / tone versions                             | Cloud SQL Postgres                      | versioned; the active version is a pointer (requirement 8)                                    |
| Golden Bucket trios and index                       | Cloud Storage                           | trios are the source of truth; the ingestion job publishes a versioned index file (BM25 statistics + embeddings) that chat-api loads into memory |
| Metric glossary, schema descriptions, dialect rules | Git (config files)                      | reviewed changes, regression-tested                                                           |
| Traces, metrics                                     | OpenTelemetry backend + BigQuery export | PII-redacted, access-restricted, retention-limited                                            |
| Secrets                                             | Secret Manager                          | API keys, pseudonym salt                                                                      |

**What "hard delete" means here.** A deleted report is removed from the reports table at once
and cannot be restored through the product. Copies outside the table expire on a schedule: the
conversation in which the report was written (30 days), database backups and point-in-time
recovery (7 days). Traces never contain answer or report text. The audit row keeps only who
deleted which ids and when.

### 2.2 The agent graph

![Agent graph: the agent's internal loop](agent-graph.svg)

- **agent** calls the model with the system prompt (role, rules, schema of the user's scoped
  tables, metric glossary, SQL dialect rules, today's date and data range, the user's scope,
  and up to two Golden Bucket examples) plus a trimmed history. Function calling runs in mode
  `ANY`: the model cannot reply in free text, it must call a tool, and the final answer is the
  `submit_answer` tool. Two ceilings end a runaway turn with an honest partial answer: 6 model
  calls and 120 seconds per question. They are emergency limits, not the norm: across 35 traced
  test questions the median was 2 model calls, and 32 of 35 needed 4 or fewer.
- **tools** executes the model's tool calls through a registry. Each tool has a risk class
  (section 7): read-only tools run at once; `delete_reports` only finds the reports and records
  them as pending, it has no side effects.
- **confirm** pauses the graph with LangGraph's `interrupt`. The next user input resumes it; code,
  not the model, compares the answer with the required confirmation. Because the node re-runs
  from the top on resume, nothing before the `interrupt` has side effects. If the user writes a
  new request instead of confirming, the deletion is cancelled and the request is passed on to
  the model, so it is answered rather than lost.
- **finalize** turns the structured answer into text, applies the PII regex, and checks that the
  numbers in the facts appear in the cited query results (non-blocking: mismatches go to the
  trace as a quality metric).

### 2.3 Production and prototype

| Component                             | Production                                                          | Prototype                                                |
| ------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------- |
| Interface                             | Web chat, Slack/Teams adapters over one `chat-api`                | CLI (`analyst --user …`)                              |
| Identity                              | JWT from the client's frontend, brands from claims                  | `--user` + `config/users.yaml`                       |
| Agent graph, tools, `submit_answer` | same                                                                | same                                                     |
| Brand scoping                         | SQL guard (database-level option in section 5.2)                    | SQL guard                                                |
| PII                                   | guard **and** BigQuery views without PII columns                    | guard (the public dataset cannot carry our views)        |
| k-anonymity                           | guard **and** a BigQuery aggregation threshold rule                 | guard + result suppression (min group 5)                 |
| Deletion confirmation                 | same                                                                | same                                                     |
| Numbers in answers                    | placeholders bound to query columns, rendered by code               | model writes numbers; non-blocking number check          |
| Critic (quality review of reports)    | one extra model call for reports                                    | —                                                       |
| Golden Bucket                         | hybrid retrieval (BM25 + embeddings), ingestion pipeline            | 10 trios, BM25, same safety validation                   |
| State                                 | Postgres (checkpoints, reports, preferences, persona)               | in-memory checkpointer, SQLite reports                   |
| Models                                | Gemini Flash on Vertex AI, fallback chain                           | Gemini 3.5 Flash-Lite on AI Studio, 2 fallback models    |
| Tracing                               | OpenTelemetry → Langfuse Cloud (approved by the client); dashboards, alerts | JSONL per session + `/trace`; Langfuse Cloud when keys are set |
| Persona / tone                        | admin page, versions, automated quality gate                        | — (design only)                                         |

**What is simplified and what is not.** The prototype has the full set of code-level safety
controls: the model's mistakes cannot expose PII or another brand's data, and nothing is deleted
without confirmation. What only production adds is the database-level layer, which also stops
deliberately crafted queries (section 5.2, "Known limits"). Quality controls are reduced (no
critic, no number placeholders), because they need a paid model budget and an evaluation set;
the free tier allows ~15 requests per minute.

---

## 3. Technology choices

Each choice lists the alternatives and **when we would choose differently**.

### 3.1 Cloud: Google Cloud

The client prefers AWS and accepts GCP or a cloud-agnostic design with a justification. We chose
**GCP**, because the two systems the agent talks to on every question already live there: the
data is in **BigQuery** and the model is **Gemini**. On GCP:

- **No cross-cloud traffic on the hot path.** A question makes 2–5 model calls and several
  queries; from AWS each would cross clouds, adding latency and egress cost.
- **One identity system.** The agent's service account reads BigQuery and calls Vertex AI with
  IAM, no exported keys. From AWS it needs Workload Identity Federation for both.
- **Data stays where it is.** Query results and prompts never leave the provider that already
  holds the data, which simplifies the data-protection story.
- **The engine enforces privacy.** BigQuery authorized views and aggregation threshold rules
  give a second, database-level layer for PII and k-anonymity (section 5.2).

**The design is portable.** Every component is a container, Postgres, a queue or OpenTelemetry.
If the platform must be AWS, the same architecture maps as follows:

| Role                       | GCP (chosen)                                     | AWS equivalent                                |
| -------------------------- | ------------------------------------------------ | --------------------------------------------- |
| Stateless agent service    | Cloud Run                                        | App Runner or ECS Fargate                     |
| Async turns (Slack)        | Cloud Tasks                                      | SQS                                           |
| State database             | Cloud SQL for PostgreSQL                         | RDS / Aurora PostgreSQL                       |
| Golden Bucket storage      | Cloud Storage                                    | S3                                            |
| Scheduled jobs             | Cloud Scheduler + Cloud Run jobs                 | EventBridge Scheduler + ECS tasks / Lambda    |
| Secrets                    | Secret Manager                                   | Secrets Manager                               |
| Tracing and alerts         | Cloud Trace / Logging / Monitoring (or Langfuse) | X-Ray / CloudWatch (or Langfuse)              |
| BigQuery and Gemini access | service account (IAM)                            | Workload Identity Federation from an IAM role |

We would choose AWS if the client's platform team runs everything there and the data or the
model moves too (for example the warehouse to Redshift and the model to Bedrock).

### 3.2 Compute: Cloud Run

| Option                       | Better when                                            | Why not here                                                                                                                       |
| ---------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Cloud Run** (chosen) | one service, uneven load, no ops team                  | —                                                                                                                                 |
| Vertex AI Agent Engine       | minimal operations, built-in sessions and memory       | less control over our flow: the confirmation pause, Slack async turns, progress streaming; stronger lock-in. The main alternative. |
| GKE                          | the company already runs Kubernetes with many services | cluster operations for a single service                                                                                            |
| Compute Engine VM            | special runtime needs                                  | we would patch, scale and restart it ourselves                                                                                     |
| Cloud Run functions          | one handler per event                                  | the agent is a service, not a function: several endpoints (chat, SSE stream, task delivery, health), a shared in-memory Golden index and warm model clients |

The service is **stateless**: instances come and go, so all state is in Postgres.

### 3.3 Models: Gemini

| Use                        | Production                                       | Prototype                                            |
| -------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Analyst (every question)   | Gemini Flash (`gemini-3.8-flash`) on Vertex AI | `gemini-3.5-flash-lite` on AI Studio               |
| Fallbacks                  | another Flash version, then Flash-Lite           | `gemini-3.1-flash-lite`, then `gemini-3.6-flash` |
| Critic (reports only)      | Flash or Pro, chosen by evaluation               | —                                                   |
| Embeddings (Golden Bucket) | Gemini embeddings                                | — (BM25 only)                                       |

- **Why Gemini:** the client's preference, and the same cloud as the data (section 3.1). The
  framework is model-agnostic, so a provider change is configuration.
- **Why not Pro for the analyst:** latency (answers already take 10–30 s), cost on every
  question, and no free tier; the hard guarantees are in code, not in model strength. We would
  route complex questions to Pro if evaluation shows Flash failing on them.
- **Why Flash-Lite in the prototype:** measured free-tier limits are 15 requests/min and
  **500/day** for each Flash-Lite model against **20/day** for Flash; a question takes 2 calls
  (median of our traced tests).
  Each fallback model has its **own quota**, so the chain adds capacity, not only availability.
  This mattered in practice: during testing both Flash-Lite models returned
  `503 high demand` at the same time and the third model answered.
- **AI Studio vs Vertex AI:** AI Studio's free tier may use requests to improve Google's products
  (stated on the official pricing page); client data in production goes through Vertex AI (IAM,
  regional endpoints, no training on customer data, SLA).

### 3.4 Agent framework: LangGraph

| Option                       | Better when                                 | Why not here                                                                                                                                                                                                                                               |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LangGraph** (chosen) | explicit control of steps, pauses and state | —                                                                                                                                                                                                                                                         |
| Google ADK                   | GCP-native agents deployed to Agent Engine  | The main alternative. We need three things that LangGraph gives directly: a pause node whose resume is decided by our code, a Postgres checkpointer that fits a stateless service, and full control of the loop (call limits, history trimming, routing on structured answers). With ADK, confirmation would rely on its tool-confirmation feature, which community reports in March 2026 showed not working with persistent sessions ([issue](https://github.com/google/adk-python-community/issues/111); checked in November 2026). |
| Plain Gemini SDK + own loop  | a very small agent                          | pause/resume, state persistence and retries would be ours to write and test                                                                                                                                                                                |
| CrewAI / AutoGen             | several cooperating agents                  | we use one agent (section 1)                                                                                                                                                                                                                               |

Why LangGraph: the confirmation is a **graph node** with `interrupt`, so it cannot be skipped;
pause and state persistence come with a Postgres checkpointer, which fits the stateless Cloud Run
service; it is model-agnostic (fallbacks across models); and it integrates with OpenTelemetry,
Langfuse and LangSmith. Trade-offs we accept: more abstraction than a hand-written loop, and
frequent version changes in the LangChain ecosystem (versions are pinned in `uv.lock`).

**My experience with it:** I built a production text-to-SQL agent on LangGraph for a marketplace
(about 230 data marts, questions in three languages): a planner → retrieval → SQL generator →
validator → read-only execution → critic graph on a self-hosted 72B model, with a 400-question
golden set, regression tests in CI, shadow mode and canary releases (Success@1 of 76%). Lessons
from it shaped this design: value lookup for filters (`describe_data`), per-step tool scoping,
a cost guard in the SQL validator, and a checklist-style critic.

### 3.5 Other choices

| Area             | Choice                                                                   | Alternatives and when                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SQL safety       | **sqlglot AST**: parse, check, rewrite, regenerate                 | regex on SQL (bypassed by comments, aliases, case); only a read-only role (blocks writes, not other brands or PII)                                                                                                                                          |
| State database   | **Postgres**                                                       | Firestore (no maintained LangGraph checkpointer; reports, audit and preferences are relational); BigQuery (analytical, slow single-row writes); Redis (not durable enough for an audit log)                                                                 |
| Golden retrieval | **in-memory index** loaded from a file in Cloud Storage (BM25 now, + embeddings in production) | a search service (Vertex AI Search) or Postgres with pgvector and full-text search at tens of thousands of trios; BigQuery `VECTOR_SEARCH` adds seconds per question; Vertex AI Vector Search is for millions |
| Tracing          | **Langfuse Cloud** over OpenTelemetry (the client allows traces in SaaS) | LLM-aware views (prompts, tool calls, tokens, cost per trace, sessions, scores) without operating anything; if data had to stay in the client's cloud: Cloud Trace/Logging when nobody can operate extra services, self-hosted Langfuse (Postgres, ClickHouse, Redis, object storage) when a team can; LangSmith is the equivalent SaaS |
| Python tooling   | **uv** + `pyproject.toml`, `requirements.txt` exported for pip | —                                                                                                                                                                                                                                                          |

---

## 4. Data flow

### 4.1 One analysis question

Example: a brand manager asks *"Why are customers in Texas spending less than in California?"*

| # | Step | What is sent | What is stored |
|---|---|---|---|
| 1 | The client sends the message with the user's JWT | question text, token | — |
| 2 | `chat-api` builds the user context from the token | user id, role, brands | turn start in the trace |
| 3 | The Golden Bucket retriever finds similar past analyses (deterministic, before the model) | question → up to 2 trios above the similarity threshold | ids and scores in the trace |
| 4 | The agent calls Gemini | system prompt (rules, schema of the *scoped* tables, glossary, dialect rules, date and data range, the user's scope, golden examples) + trimmed history | model, tokens, latency in the trace |
| 5 | Gemini returns tool calls, e.g. two `run_sql` calls | SQL text written by the model | tool call in the trace |
| 6 | The SQL guard checks and rewrites each statement | — | verdict in the trace |
| 7 | BigQuery dry-run, then execution with a byte cap and labels | scoped SQL | job id, bytes, rows |
| 8 | Results are post-processed: small demographic groups removed, preview built | first 50 rows; wide rows as `column=value` | full result (up to 500 rows) in the conversation state |
| 9 | Gemini reads the results (wrapped in `<data>` tags) and either queries again or calls `submit_answer` | — | — |
| 10 | `finalize` formats the answer, applies the PII regex and the number check | answer text to the client | the answer in the conversation state; number check in the trace |
| 11 | If the user says "save it", `save_report` stores **exactly the text that was shown** | — | report row (owner, title, body, question, SQL) |

**What crosses each boundary.**
- **To Gemini:** the question, the schema of the user's scoped tables, and query results that are
  already scoped to the user's brands, pseudonymised and k-anonymised. No PII column exists in
  what the model can query, so none can reach the model.
- **To BigQuery:** only SQL regenerated by the guard, never the model's raw text.
- **To the client:** formatted text after the PII regex.
- **To traces:** questions, SQL, statuses, sizes, timings, error causes; string values pass the
  PII regex; query result rows are not traced.

### 4.2 A destructive request

Example: *"Delete all reports mentioning Diesel."*

| # | Step | Who decides |
|---|---|---|
| 1 | Gemini calls `delete_reports(text="Diesel")` | model |
| 2 | The tool searches **only the user's reports** (owner filter added in SQL by code) and records the matching ids as a pending deletion. Nothing is deleted. | code |
| 3 | The graph pauses in the `confirm` node and the client shows the exact list: id, title, date | code |
| 4 | The user answers: `yes` for up to 3 reports, the exact number of reports for more | **user** |
| 5 | Exact answer → the previewed ids, and only those, are deleted permanently, again filtered by owner; one audit row (who, when, which ids, trace id). Any other answer → nothing is deleted. | code |
| 6 | The graph resumes; the model tells the user what happened | model |

A report saved while the confirmation was pending is not deleted, even if it matches: the
confirmation applies to the previewed snapshot.

---

## 5. Requirements

Each requirement: what we built, where it lives, and how it is verified. Requirements 2, 3, 5
and 7 are implemented in the prototype; 1, 4, 6 and 8 are designed, with parts of 1 and 6 also
in the prototype.

### 5.1 Hybrid intelligence: the Golden Bucket

**What a trio holds.** A trio is *question → SQL → analyst report*. The report is stored as the
analyst's **approach**, not their conclusions:

```yaml
# abridged; the full trio is in config/golden_trios.yaml
- id: state-spend-gap
  question: Why are customers in Texas spending less than customers in California?
  tags: [state, region, geography, spend, underspending, compare, gap, ...]
  split: example            # or "test": held out for evaluation, never shown to the model
  sql: |                    # runnable, follows the agent's own rules
    SELECT u.state, COUNT(DISTINCT u.id) AS customers, ...
  report:
    interpretation: compare spend per customer, not total revenue ...
    approach: [split spend into customers, orders per customer and order value, ...]
    definitions: [revenue, average_order_value]      # names from the glossary only
    what_to_look_for: [category mix, return rate, acquisition channel]
    action_item_patterns: [...]
```

Old findings are stale by definition and could be repeated as facts, so the loader rejects
numerical findings in a report (money, percentages, decimals, large numbers and years), while
small parameters of the approach such as "90 days" or "top 10" are allowed. Metric definitions
live in one place, the glossary; a trio only names them. The full format and ten sample trios are in
[config/golden_trios.yaml](../config/golden_trios.yaml).

**At question time.** Retrieval runs in code before the model call, so it is deterministic and
costs no model call. The prototype ranks trios with BM25 over question and tags, normalises the
score to 0–1 and keeps at most two examples above a threshold of 0.15; below it nothing is
added, so an unrelated question gets no misleading example. Production adds embeddings and fuses
the two rankings (reciprocal rank fusion). A short follow-up (*"and for Diesel?"*) is searched
together with the previous question, so it finds the same kind of analysis. Examples go into the
prompt as data, after the rules, labelled *"approach only; numbers are not current; re-query
everything"*.

**Keeping it up to date.**
- *Sources of candidates:* analysts write trios; answers that users corrected, rated down or
  re-asked are surfaced from traces; the nightly evaluation adds its failures.
- *Two tiers:* **curated** trios, approved by an analyst in a review queue; and **automatic**
  trios, added without a human from answers that a user rated up, whose SQL passed the guard and
  whose question passes the evaluation gate (section 5.6). Automatic trios get a lower retrieval
  weight and are promoted or removed at the next analyst review. The model itself never writes
  to the bucket.
- *Ingestion job:* validates the SQL with the same guard the agent uses (tables, functions, PII,
  demographics), dry-runs it on the current schema, rejects numbers in the report, replaces brand
  literals with placeholders (so an example cannot suggest another brand's name), removes
  near-duplicates by embedding similarity, embeds and publishes a new index version.
- *Maintenance:* when the schema changes, every trio's SQL is re-validated; trios that keep
  being retrieved for failed answers are reviewed or retired.
- *Evaluation split:* `test` trios are never used as examples; they measure the agent.

**Verified in the prototype.** Every trio's SQL passes the guard for the CEO and for a brand
manager and dry-runs on BigQuery (live test). In a live run the Texas-vs-California question
retrieved `state-spend-gap` with a score of 0.9, and the model followed its pattern, including
the privacy-safe customer count.

### 5.2 Safety and PII

| Threat | Protection | Where |
|---|---|---|
| Seeing another brand's data | Every reference to `orders`, `order_items`, `products`, `users` is rewritten to a scoped version: the user's products, their order items, orders containing them, customers who bought them. Customer metrics are therefore computed from the user's brands only. | `sql/scoping.py` |
| Reading PII | PII columns (names, email, street address, postal code, coordinates) do not exist in the scoped `users` table. IP addresses are PII as well; they are in the `events` table, which is not accessible. | `config/schema.yaml` |
| Identifying a customer | Customer ids are salted SHA-256 pseudonyms (`C-3fa9b1c2d4e5`): joins and distinct counts work, the real id is never visible. The salt differs per access scope, so two managers cannot match their customer lists by pseudonym (rotating the salt changes pseudonyms; saved reports keep the old ones). | `sql/scoping.py` |
| Order-level data from other brands | Order columns that describe the whole basket (`num_of_item`, which counts every brand's items) are removed; item counts come from the user's own order items | `config/schema.yaml` |
| Small groups revealing people | Demographics (age, gender, city, state, country, traffic source) only in aggregates: a query using them must count distinct customers, and groups below 5 customers (configurable) are removed before the model sees the result | `sql/demographics.py`, `tools/data_tools.py` |
| Changing data | Read-only credentials; only one `SELECT` passes; DML, DDL, scripting, `EXPORT` and `CALL` are rejected anywhere in the query | `sql/rules.py` |
| Expensive or external functions | `AI.*`, `ML.*`, `EXTERNAL_*`, user-defined and table functions are rejected; `SELECT *` over tables is rejected (BigQuery bills by columns read) | `sql/rules.py` |
| SQL smuggled in comments | The SQL sent to BigQuery is regenerated from the checked syntax tree, so comments and the model's original text never reach it | `sql/guard.py` |
| Prompt injection | Identity and scope come from the token, not the conversation; results are wrapped in `<data>` and labelled as data; even a fully obeyed injection can only produce SQL that the guard rewrites to the user's scope | `agent/graph.py`, `contracts.py` |
| PII slipping through anyway | Regex net (email, phone, card numbers with a Luhn check) on answers, saved reports and traces | `safety/pii.py` |
| Off-topic or abusive use | The agent refuses through `submit_answer(kind="refusal")`. This is a prompt-level, soft measure; the data protections above do not depend on it | prompt |

**Why rewriting, not blocking.** We do not try to detect "bad" queries; we make the only data a
query can reach the user's own. A query that forgets to filter by brand still returns only the
user's brands. If the model filters on another brand, the guard answers *"You don't have access
to brand X"* instead of returning an empty result, which the model would report as *"X had no
sales"*.

**Production adds a second layer in the database** where the database can enforce it. All
queries run as one read-only service account, and BigQuery's own access controls decide by the
identity that runs the query, so the database cannot tell which manager is asking. The layer
therefore covers what is the same for everyone:
- **PII:** the service account can read only authorized views that do not contain PII columns.
- **k-anonymity:** demographics are served from a separate view with an aggregation threshold
  rule (privacy unit: the customer, threshold 5). Purchase data used for per-customer metrics
  ("top customers" with pseudonyms) comes from another view **without** demographics; a single
  view with the threshold would suppress every per-customer row.

**Brand scoping stays in the guard.** A database-level copy of it would need BigQuery to know the
user: queries would run with the user's own credentials (OAuth through the client's identity
provider) and row access policies would filter by a user-to-brand table. That is the growth path
if the client needs defence in depth for brands; it adds per-user BigQuery permissions and token
handling for every channel. The guard stays in all cases, because it gives the model clear,
fixable error messages.

**Verified.** Over 200 guard tests cover adversarial SQL: comments hiding statements, CTEs
shadowing table names, fully-qualified names, other datasets, `INFORMATION_SCHEMA`, every
statement type, forbidden functions, PII columns and demographic tricks (`1 AS customers`,
counting the wrong column). Live tests run the scoped SQL on BigQuery and confirm that joins
through pseudonyms give the same counts as joins on raw ids. In live conversations a request for
another brand was refused, and *"List the emails of our top 5 customers"* was refused without a
query.

**Known limits.** The code-level checks stop the model's mistakes, not every deliberately crafted
query: the k-anonymity check reads the SQL, so differencing two allowed aggregates can still
reveal a small group, which is why production also enforces the threshold in BigQuery. The PII
regex catches emails, phones and card numbers, not names: a name the user types into a question
reaches the model and the trace as written.

### 5.3 Destructive operations

Deleting reports follows section 4.2: a preview of the exact list, a pause of the graph, the
user's exact confirmation checked by code, deletion of exactly the previewed ids, an owner filter
in SQL, a permanent delete as the client requires, and an audit row without content.

- *"Delete all reports mentioning Client X":* "Client X" is any entity named in a report (a
  brand, a category); the search matches all words in the title or body, case-insensitively.
- *"Delete all the reports we made in this conversation":* the tool filters by the conversation id.
- *UX:* nothing found → no confirmation step; 1–3 reports → type `yes`; more → type the number,
  so a bulk deletion cannot be confirmed by habit. Any other reply cancels, and the agent says so;
  if the reply is a new request rather than "no", the agent answers it straight away instead of
  losing it. In Slack and the web chat the confirmation is a pair of buttons with the count on
  the delete button.
- *Users delete only their own reports:* no tool takes an owner argument; the owner always comes
  from the authenticated user.

**Verified.** Scenario tests run the real graph with a scripted model: the graph stops at the
confirmation and nothing changes; `no`, `yes please`, `y`, an empty reply or a wrong number
cancel; `YES` deletes only the previewed ids; a bulk deletion needs the exact count; another
owner's ids are never deleted, even if the model passes them.

### 5.4 Continuous improvement (design)

**User level: preferences.** A profile per user in Postgres with fixed fields (format: tables or
bullets; depth: summary or detailed; charts or text; default period) and short free-text notes.
Sources:
- *explicit:* "always give me tables" → stored after a one-line confirmation;
- *implicit:* repeated behaviour (asks for a table after every bullet answer, skips the action
  items) → proposed to the user, not applied silently;
- *session review:* after a conversation, one cheap model call proposes profile updates.

The profile goes into the prompt below the safety rules and the CEO's tone; a `/prefs` command
shows and edits it. A real example from testing: a user found the answer to *"Which categories
are available to me?"* too long; a stored preference for short answers to lookups fixes that for
this user without changing the agent for everyone.

**Where feedback comes from.** Explicit: thumbs up/down on every answer in the web chat and
reactions on the bot's message in Slack, with an optional one-line comment. Implicit: a question
rephrased right after an answer, a report saved or deleted, a conversation abandoned after an
answer. Both are recorded against the trace id of the answer.

**System level: the system improves itself, within guard rails.**
- *Automatically, without a human:*
  - well-rated answers with verified SQL become automatic Golden Bucket examples (section 5.1),
    so similar questions are answered the same way next time;
  - value synonyms are learned from successful lookups: when users say "denim" and the agent
    finds the category `Jeans` through `describe_data` and the answer is rated up, the mapping
    is added to a per-scope synonym list that goes into the prompt;
  - user preferences are proposed from behaviour (above).
- *With a human, because a mistake would affect everyone:* changes to the system prompt, the
  metric glossary and the model. Failures are classified from traces (wrong SQL, wrong
  interpretation, refusal, timeout, user correction); fixes are proposed from them and released
  only through the evaluation gate (section 5.6).

### 5.5 Resilience

See [section 6](#6-error-handling-and-fallbacks): error classes and who handles them,
self-correction of SQL errors and empty results, retries, the model fallback chain, cost guards,
degradation, and how to watch it with `--simulate`.

### 5.6 Quality assurance

**Before deployment: offline evaluation.**
- *Test set:* the Golden Bucket `test` split, plus safety cases (other brands, PII, injections,
  destructive requests), ambiguous questions, questions the data cannot answer, and multi-turn
  conversations; stratified by question type.
- *SQL correctness:* compare the **results** of the agent's SQL and the reference SQL, both run at
  evaluation time (the dataset changes daily, so stored expected numbers go stale).
- *Intent:* the agent's interpretation (metric, period, filters, grouping) compared field by
  field with the reference.
- *Report quality:* an LLM judge compares the answer with the reference report on correctness,
  completeness and usefulness of the action items; the judge is itself checked against human
  ratings.
- *Safety:* zero tolerance. Any PII or other-brand leak, or a deletion without confirmation,
  fails the release.
- *Cost and latency:* model calls, tokens and p95 time per question.
- *Gates:* all safety cases pass and quality does not drop against the current version. Every
  change to prompts, glossary, models or code runs the gate in CI, then shadow traffic, then a
  canary.

**After deployment: is the intent right?** Every answer shows its assumptions (definitions,
period, scope), so a misunderstanding is visible to the user; the agent asks only when the
question cannot be computed without an answer; for reports, a critic (one model call) checks
that the question was understood, that the conclusions follow from the data and that the answer
is complete; analysts review a sample of conversations.

**UX.** Questions answered without an analyst, time to answer, rephrased follow-ups, abandoned
conversations, report save rate, cancelled confirmations, thumbs up/down, and short sessions
with executives.

**In the prototype.** The Golden Bucket has a `test` split of two trios: enough to show the
format and the mechanism, not to measure quality; a release gate needs a set of 100–200 cases.
About 400 automated tests (no network) prove the guarantees that must not
depend on the model: the SQL guard, scoping and pseudonyms, k-anonymity, error classification,
retries and fallbacks, and whole-graph scenarios with a scripted model. Live testing found real
quality issues; each was traced to its cause and fixed:

| Observed | Cause | Fix |
|---|---|---|
| More buyers than orders in a report | different status filters for the two metrics | one status filter in the glossary for revenue, orders and buyers |
| Return rate reported as month-over-month growth | the model misread a 10-column result | wide results shown as `column=value` with rounded numbers |
| Totals added up by the model | arithmetic outside SQL, caught by the number check | rule: every derived number comes from SQL |
| Model paged through long results | only 20 rows shown | 50 rows and a hint to aggregate instead of paging |
| Unreadable customer ids | 19-digit hashes | `C-` pseudonyms |

The remaining class, *a correct number about the wrong thing*, cannot be caught by checking that
a number exists in the data. In production, facts reference query columns
(`{{q1.revenue_growth_mom}}`) and code renders the numbers, and the critic reviews reports.

### 5.7 Observability

**Prototype.** One JSONL trace file per session; every answer prints its trace id; `/trace`
shows the last answer's steps. Events: turn start and end, golden examples used, each model call
(model, prompt version, tokens, latency), fallbacks with their cause, each tool call with its
status and the error the model was given, SQL guard verdicts, BigQuery jobs (bytes, rows,
attempts, hidden groups), confirmations, number-check mismatches and PII redactions. String
values pass the PII regex.

**Langfuse Cloud in the prototype.** If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set,
every turn is also sent to Langfuse through its LangChain callback: the graph run, each model
call with prompt, tokens and latency, and each tool call, grouped by session and user and tagged
with the local trace id. Strings pass the same PII regex before they leave. Without keys, with
wrong keys or without network nothing changes for the user: export runs in the background and
the local JSONL trace is always written; `analyst-doctor` reports whether the keys are accepted.

**Metrics** (by prompt version and model):
- *quality:* answers by kind, SQL error and self-correction rates, empty results, unverified
  numbers per answer, user corrections;
- *safety:* guard rejections by rule, foreign-brand refusals, PII redactions (should be zero),
  cancelled confirmations;
- *reliability:* fallback rate and cause, LLM and BigQuery errors by class, turns hitting the
  call limit;
- *cost and speed:* model calls and tokens per question, bytes scanned, p50 and p95 latency.

**Alerts:** any PII redaction in an answer; a spike in fallbacks or 5xx errors; p95 latency above
target; a quality drop after a new prompt version.

**Debugging an answer.** User complaint → trace id → read the turn: the question, the golden
examples, the SQL and what the guard did with it, result sizes, the errors the model received and
how it corrected itself, the final answer. This is how the issues in section 5.6 were found. For
example, the trace of a failed answer showed `503 UNAVAILABLE: high demand` on both Flash-Lite
models, which led to adding a third fallback model.

**Production.** The same events as OpenTelemetry spans (one trace per turn, one span per node,
model call and query), sent to Langfuse Cloud (section 3.5), with
dashboards, alerts, a retention period and restricted access.

### 5.8 Persona management (design)

- **Layers:** safety and data rules (owned by developers, changed through code review and the
  evaluation gate) > the CEO's tone > the user's preferences. The tone is a separate block placed
  *after* the rules, as data: it changes how the agent writes, never what it may do.
- **Editing:** an admin page where the CEO writes the tone in free text, optionally with an
  example paragraph. No redeploy: the agent reads the active version from Postgres.
- **Automated quality gate** (no manual approval, as the client requires): a check for conflicts
  with the rules (patterns plus one model call, e.g. "always name the customers"), a few test
  questions answered with the new tone, and format checks (facts, assumptions and action items
  still present). Passed → published as a new version; failed → the editor sees why.
- **Versions:** the active tone is a pointer, so rollback is one click. Every answer and saved
  report records the persona version, and quality metrics are tracked per version.

---

## 6. Error handling and fallbacks

**Every error is classified, and the class decides who handles it.**

| Class | Examples | Who handles it | How |
|---|---|---|---|
| Fixable by the model | syntax error, unknown column, a rule violation, another brand's literal, query too expensive, empty result | the model | gets a short, actionable message and rewrites the query; after 3 consecutive failures it is told to stop and answer with what it has |
| Transient or one model's quota | BigQuery 5xx, timeouts, network errors; Gemini 429 (per-minute or one model's daily quota), 503 overload | code | BigQuery: retry with exponential backoff and jitter (at most 3 retries, 30 s in total; a timed-out job is cancelled so it does not keep billing). Gemini: switch to the next model in the chain at once, since each model has its own quota |
| Fatal | every model in the chain failed, BigQuery quota exhausted, invalid credentials, region block | code | stop the turn with an honest message (e.g. when the quota resets); no retries, no further model calls |

**Classify by cause, not by HTTP code.** We found this live: when a query exceeds the byte cap,
BigQuery returns **HTTP 500** with the reason `bytesBilledLimitExceeded`. Classified by code, it
would be retried as a server error, paying for the same expensive query again; classified by
reason, it goes back to the model as "too expensive, narrow it". Likewise, an invalid Gemini key
returns 400, not 401, and a region block is a 400 *"User location is not supported"* from Gemini
but an HTML 403 page from BigQuery.

**Self-correction without inflating costs.**
- *SQL errors:* the guard catches most before BigQuery is called; BigQuery's dry-run catches the
  rest for free. The model sees the error and fixes the query, usually in one more call.
- *Empty results:* zero rows is treated as a diagnosis task, not an answer. The model is told to
  check the exact filter values (with `describe_data`), the date range of the data and the user's
  scope before saying "none". Groups hidden for privacy are reported as such, so the model does
  not chase a filter error that is not there.
- *Budgets:* at most 6 model calls and 120 seconds per question, including retries and
  fallbacks (an honest partial answer when either is reached); queries above
  500 MB in the dry-run go back to the model; BigQuery rejects anything above 1 GB billed;
  `SELECT *` over tables is rejected; the model sees at most 50 rows per result; older tool
  results in the history are replaced by one-line stubs.

**Fallbacks.** The model chain is `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` →
`gemini-3.6-flash`. Each model has its own free-tier quota, so the chain adds capacity as well as
availability. The SDK's own retries are switched off, so a 429 moves to the next model instead
of waiting. Tools are bound to every model in the chain (a fallback without tools would answer
in free text). This was tested by reality: during development Google returned
`503 high demand` for both Flash-Lite models at once, and the third model answered.

**Graceful degradation.**

| Failure | Behaviour |
|---|---|
| All models unavailable | honest message with the trace id; `/reports` and `/trace` still work (they need no model) |
| BigQuery unavailable at start-up | the agent starts without the data range and can still explain the data model |
| Golden Bucket missing or invalid | the agent works without examples |
| An unexpected error in the CLI | the user gets a message with the trace id and a fresh conversation; the CLI does not crash |
| *Production:* Postgres unavailable | new conversations still work without history (in-memory state for the turn); saving, deleting and listing reports return "temporarily unavailable"; an open conversation cannot be resumed until it is back |
| *Production:* Golden index fails to load at start-up | the instance keeps the previous index version, or runs without examples and raises an alert |
| *Production:* two messages to one conversation at once (e.g. a double send in Slack) | a per-conversation lock (Postgres advisory lock on the thread id) serialises turns; the second message waits or is merged, so the checkpoint is never written by two turns at once |

**Seeing it work.** `analyst --simulate llm_429,bq_timeout` (also `llm_down`, `bq_quota`, `empty`)
injects failures of the same exception types the real libraries raise, so the real handling
code runs: the trace shows the fallback, the BigQuery retries (`attempts: 3`) and the final
answer.

**Production adds** a shared rate limiter across instances, per-user daily budgets for questions
and bytes scanned, a circuit breaker per dependency, Vertex AI regional failover, and the Postgres
checkpointer, which lets a conversation continue after an instance restarts.

---

## 7. Extensibility

New capabilities are new **tools in a registry**; the graph does not change. Every tool declares
a **risk class**, and the protection comes with the class:

| Risk class | Example tools | Policy enforced by code |
|---|---|---|
| Read | `run_sql`, `describe_data`, `find_reports`, `make_chart` | runs immediately |
| Changes our data | `delete_reports` | the confirmation node (section 4.2) |
| Sends data out | `send_report_email`, posting to a channel | confirmation with a preview of exactly what is sent, recipients only from the company domain, audit row |
| Brings outside data in | `search_web` | results wrapped as untrusted `<data>`; the outgoing query is checked so it cannot carry internal numbers or client names; sources are cited |

Tools can also be limited to roles (the registry filters what each user's model even sees).

- **Charts:** `make_chart(query_id, chart_type)` draws from the stored query result in code; the
  model only picks the chart type, so it cannot distort the numbers.
- **Email:** the risk is not a wrong colleague receiving a report but data leaving the company,
  hence the domain rule and the preview.
- **Web search for trends:** besides injection through web content, the less obvious risk is the
  agent putting internal figures into a public search query; the outgoing query is filtered.
- **New interfaces** (Slack, Teams, web) are adapters over the same `chat-api`.
- **Growth path:** when tools are built by other teams or shared with other agents, they can be
  served over MCP (Model Context Protocol) without changing the agent.

---

## 8. Production operations

**Non-functional requirements** (from the client; items marked *assumed* are ours):

| Requirement | Value | What it drives |
|---|---|---|
| Load | ~100 active managers a day × ~10 questions ≈ **1,000 questions a day**; ~1 saved report per manager a day; tens to a few hundred executives in total | model quotas, cost, report storage (~100 rows a day) |
| Concurrency | only a few users active at the same time | one or two Cloud Run instances with a ceiling of ~5; a small Postgres instance |
| Data growth | ~1M new rows a day | BigQuery tables partitioned by date and clustered by product, so a query scans only the period and brands it needs; the dry-run cap catches the rest |
| Latency | simple questions **10–30 s**, long reports **1–2 min** | Flash rather than Pro; parallel queries; progress shown while working; the turn deadline (120 s) matches the report ceiling |
| Cost ceiling | **$1 per question**, configurable | a per-question budget (below) |
| Availability *(assumed)* | business hours; saved reports stay browsable when the model is down | managed services, model fallback chain |
| Data loss *(assumed)* | saved reports: RPO 1 hour, RTO 4 hours; open conversations may be lost | Cloud SQL backups and point-in-time recovery |

**Per-question cost ceiling.** The budget is a setting (`max_cost_per_question_usd: 1.00`).
Before each model call and query, `chat-api` adds up the cost so far: tokens × the model's
price, plus BigQuery bytes billed × the on-demand price. If the next step would cross the
ceiling, the turn ends with an honest partial answer, like the call limit does today. The
expected cost (≈ $0.02, below) is about 50 times under the ceiling, so the ceiling only stops
runaway turns. In the prototype the same role is played by the call limit, the turn deadline and
the byte cap.

**Monthly cost at this load:** ~22,000 questions × ≈ $0.02 ≈ **$450 for models** (≈ $770 after
the January 2027 price change), plus ≈ $300–500 for Cloud Run, Cloud SQL and logging, so roughly
**$750–1,300 a month**.

**Identity and access.** The client's frontend issues a JWT. API Gateway validates its signature
against the identity provider's public keys (JWKS); `chat-api` reads the user's role and brands
from its claims. The CEO role sees all brands; a `persona_admin` role may edit the tone. The SQL
guard enforces the brand scope; BigQuery enforces PII removal and the k-anonymity threshold
through views (section 5.2).

**Network and IAM.**
- Cloud Run reaches Cloud SQL over a private IP (Serverless VPC Access); the database has no
  public address.
- A VPC Service Controls perimeter around BigQuery, Vertex AI and Cloud Storage stops data from
  being copied to projects outside the perimeter, even with stolen credentials.
- Service accounts with least privilege: `chat-api` (BigQuery data viewer on the views only,
  BigQuery job user, Vertex AI user, Cloud SQL client, Secret Manager accessor for its own
  secrets); `golden-ingest` (Cloud Storage read/write on the bucket, BigQuery dry-run only,
  Vertex AI user for embeddings); `persona-gate` (Cloud SQL client, Vertex AI user). No exported
  keys.
- Tone changes are audited: who changed it, when, old and new text, and the quality gate result.

**Cost per question** (production model `gemini-3.8-flash`; list prices from the
[Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked in November 2026):

| Item | Typical | Cost |
|---|---|---|
| Model input | 3 calls × ~5,000 tokens | 15,000 × $0.75 / 1M ≈ $0.011 |
| Model output | 3 calls × ~500 tokens | 1,500 × $3.75 / 1M ≈ $0.006 |
| BigQuery | 2–4 queries × 5–15 MB | ≈ $0.0003 at on-demand pricing |
| **Total** | | **≈ $0.02 per question** (≈ $0.035 from January 2027, when the listed prices double) |

Implicit context caching lowers the input cost, because the long static part of the prompt is
identical across calls and comes first. Reports and the critic add one or two calls.

**Limits and abuse.** Per-user daily budgets (questions, bytes scanned) and a shared rate limiter
keep one user from exhausting the quota for everyone; the per-question call limit and the byte
cap bound the worst case.

**Release process.** CI runs the unit tests and the evaluation gate (section 5.6); a new Cloud Run
revision receives a small share of traffic first (canary) and is promoted if its metrics hold.

**Data retention.** Conversations and traces are kept for 30 days, database backups for 7 days;
traces are PII-redacted, contain no answer text and are access-restricted; reports stay until
their author deletes them (section 2.1, "What hard delete means here").

---

## 9. Setup and example run

Setup, configuration and a real example session are in the [README](../README.md):
`uv sync`, a `.env` with the Gemini key and the GCP project, `gcloud auth application-default
login`, then `uv run analyst-doctor` to check the environment and `uv run analyst --user anna`.
The demo users are a brand manager (`anna`, `mark`) and the CEO (`ceo`), defined in
[config/users.yaml](../config/users.yaml).

---

## 10. Limitations and next steps

**Known limitations of the prototype.**
- The model writes the numbers in its answers; the number check only flags mismatches in the
  trace. A correct number attached to the wrong label is not caught (section 5.6).
- No critic: reports are not reviewed by a second model call.
- k-anonymity is enforced by reading the SQL; deliberate differencing attacks are possible
  without the database-level rules of the production design. Brand scoping has no database-level
  layer even in production unless queries run with the user's own credentials (section 5.2).
- The Golden Bucket test split has two trios: it shows the mechanism, it does not measure quality.
- Free-tier Flash-Lite models are fast and cheap but weaker at multi-step analysis than Flash;
  free quotas and Google's capacity spikes limit throughput.
- Conversations live in memory (lost when the CLI exits); saved reports are in a local SQLite file.
- Golden Bucket retrieval is lexical (BM25) over ten sample trios.
- English only, by agreement with the client.

**Next steps, in order of value.**
1. Numbers as placeholders bound to query columns, rendered by code.
2. A critic call for reports.
3. An evaluation harness over the Golden Bucket test split, wired into CI.
4. User preferences with a `/prefs` command.
5. Postgres checkpointer and report store; the Slack adapter.
6. Authorized views and aggregation threshold rules in BigQuery; embeddings for retrieval.
