# Retail Analyst Agent

A command-line data-analysis agent for retail executives. It answers questions about sales,
customers and products in BigQuery's public `thelook_ecommerce` dataset, writes short reports
with action items, and keeps a library of saved reports.

Built with **Gemini** (function calling) and **LangGraph**. One analyst model plans, writes SQL and
interprets the results; every protection is enforced by code around it, not by the prompt:

- **Per-user data access and PII** – every query is parsed (sqlglot) and rewritten so that each
  table only contains the user's products, customers have no personal-data columns, and customer
  ids are salted hashes. Only single read-only `SELECT`s over the four allowed tables pass.
- **Destructive actions need the user, not the model** – deleting reports pauses the graph
  (`interrupt`) and shows a preview; only the exact confirmation deletes, and only those reports.
- **Resilience without cost blow-ups** – dry-run cost check, hard byte cap per query, retries only
  for transient errors, a fallback model with its own quota, a per-question call limit.
- **Observability** – every step of every answer is traced to JSONL and viewable with `/trace`.

The design (architecture, decisions and trade-offs, production deployment) is in
[docs/DESIGN.md](docs/DESIGN.md).

## Quick start

**Prerequisites**
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and the dependencies for you)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com) (free tier is enough)
- A Google Cloud project for BigQuery (the free [BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) is enough)
  and the [gcloud CLI](https://cloud.google.com/sdk/docs/install)

**Setup**
```bash
git clone https://github.com/ZhuIik/retail-analyst-agent.git
cd retail-analyst-agent
uv sync

cp .env.example .env            # then set GOOGLE_API_KEY and GCP_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID

uv run analyst-doctor           # checks the key, the models, BigQuery access and the dataset
```

Without uv (Python 3.11–3.13): `python -m venv .venv`, activate it, `pip install -r requirements.txt`
(pinned versions, installs this package too), then run the same commands without `uv run`.

**Run**
```bash
uv run analyst --user anna      # brand manager: Calvin Klein, Diesel, Carhartt
uv run analyst --user mark      # brand manager: Levi's, True Religion, Columbia
uv run analyst --user ceo       # all data
```
`--user` simulates login; users and their data scope are in [config/users.yaml](config/users.yaml).

In the chat: `/help` (examples), `/reports` (your saved reports), `/trace` (what happened in the
last answer), `/quit`. These commands work even when the model is unavailable.

**Optional: traces in Langfuse Cloud.** Every answer is traced locally (`traces/*.jsonl`,
`/trace`). To also see traces in [Langfuse](https://cloud.langfuse.com), set
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env`. Without them the
agent works exactly the same; wrong keys or no network never interrupt a conversation.

**See the failure handling** (requirement 5) by injecting faults:
```bash
uv run analyst --user ceo --simulate llm_429,bq_timeout
```
| Fault | What you will see |
|---|---|
| `llm_429` | the primary model is rate-limited; every call goes to the fallback model |
| `llm_down` | all models fail; an honest message instead of a crash |
| `bq_timeout` | the first BigQuery attempts time out; the query is retried by code (`attempts` in `/trace`) |
| `bq_quota` | BigQuery quota exhausted; the turn stops with a clear message, no retries |
| `empty` | queries return no rows; the agent diagnoses filters instead of reporting "0" |

## Example session

Real output from 11 November 2026, user `anna` (`...` marks omitted lines). The first query
failed and the agent corrected it on its own; the totals come from a separate SQL query.

```text
you> Give me a short report on revenue by month for the last 3 full months with totals, by brand.
· querying: Revenue by month for the last 3 full months by brand for Anna's brands (Calvin Klein,
Diesel, Carhartt)
· status=error error=invalid_sql
· querying: Revenue by month for the last 3 full months by brand for Anna's brands
· querying: Total revenue by month for the last 3 full months across all brands in scope
· writing the answer
Over the last 3 full months (August through October 2026), total portfolio revenue remained stable
between $18,616 and $19,300 per month. Calvin Klein consistently led in revenue and unit volume
across all three months, followed by Diesel and Carhartt.

Facts
 • In October 2026, revenue by brand was: Calvin Klein $7,348.40 (124 units), Carhartt $6,488.47
   (98 units), Diesel $4,779.44 (42 units). [q1]
 ...
 • Total revenue across all three brands was $19,298.96 in August 2026, $18,812.84 in September
   2026, and $18,616.31 in October 2026. [q2]

Hypotheses (possible explanations, not verified by the data)
 • ... individual brand contributions fluctuated (e.g., Diesel dropped in October while Calvin
   Klein and Carhartt grew).

Action items
 • Monitor Diesel's sales trend to see if the October dip is temporary or part of a wider softening
   in demand.
 • Capitalize on Calvin Klein's steady performance with targeted seasonal promotions ahead of the
   holidays.

Assumptions
 • Last 3 full months refers to August 2026, September 2026, and October 2026 (today is November
   11, 2026).
 • Revenue excludes cancelled and returned items.
 • Brand scope is restricted to Calvin Klein, Diesel, and Carhartt.

Save this as a report "Revenue by Month and Brand (Aug-Oct 2026)"? (yes/no)
trace a3c3300045ae

you> yes
· saving report 'Revenue by Month and Brand (Aug-Oct 2026)'
· writing the answer
I have saved this report for you.

you> Show sales for Nike last month
· looking up brand values
· writing the answer
I cannot show sales for Nike because your data scope is limited to Calvin Klein, Diesel, and
Carhartt. Nike products are outside your brand management scope.
...

you> Delete my report about monthly revenue
· finding reports to delete
                          Reports to delete
┌────┬───────────────────────────────────────────┬──────────────────┐
│ id │ title                                     │ created          │
├────┼───────────────────────────────────────────┼──────────────────┤
│ 1  │ Revenue by Month and Brand (Aug-Oct 2026) │ 2026-11-11T13:32 │
└────┴───────────────────────────────────────────┴──────────────────┘
Deletion is permanent. Type yes to delete; anything else cancels.
you> yes
· writing the answer
I have permanently deleted your report about monthly revenue.

Facts
 • Permanently deleted report #1: 'Revenue by Month and Brand (Aug-Oct 2026)'.
...
```

`/trace` after any answer shows its steps: golden examples used, each model call, the SQL and
the guard's verdict, BigQuery bytes and rows, retries and fallbacks.

Asked by the CEO to "List the emails of our top 5 customers by spend", the agent refused without
running a query. Even if it had tried, the scoped `users` table has no email column.

## Tests

```bash
uv run pytest -m "not live"   # ~400 tests, no network or credentials needed
uv run pytest -m live         # live checks against BigQuery (skipped without credentials)
```
The unit tests cover the guarantees that must not depend on the model: the SQL guard against
adversarial queries (DML, comments, CTE shadowing, forbidden functions, PII columns, other
tables), scoping and pseudonyms, k-anonymity, the Golden Bucket format, error classification, retries and fallbacks, and full graph
scenarios with a scripted model, e.g. *no deletion happens without the exact confirmation*.

## Project layout

```text
config/                 everything that changes without code changes
  users.yaml            users and their brands (stand-in for the JWT the frontend sends)
  schema.yaml           tables the model sees; PII removed, ids pseudonymised, demographics marked
  glossary.yaml         metric definitions (revenue, churn, ...)
  golden_trios.yaml     Golden Bucket: Question -> SQL -> analyst approach (sample set)
  settings.yaml         models, limits, thresholds
src/retail_analyst/
  cli.py                chat loop, progress lines, confirmation prompt, /reports /trace
  agent/graph.py        LangGraph: agent -> tools -> confirm -> finalize
  agent/prompts.py      system prompt (static part first for implicit caching)
  agent/answer.py       answer formatting and the non-blocking number check
  sql/                  SQL guard: rules, access check, scoping rewrite, k-anonymity
  tools/                tool registry (with risk classes), data tools, report tools
  golden/               Golden Bucket loader and BM25 retriever
  bq/client.py          BigQuery: dry-run, byte cap, error classification, retries
  llm.py                Gemini models, fallback chain, rate limiter
  reports/store.py      saved reports (SQLite), owner filter, permanent delete, audit log
  tracing.py            JSONL traces
  faults.py             fault injection for --simulate
  doctor.py             environment check
tests/
```

## Configuration notes

- **Models**: `gemini-3.5-flash-lite`, falling back to `gemini-3.1-flash-lite` and then
  `gemini-3.6-flash` (separate free-tier quotas: 500/day for each Flash-Lite, 20/day for Flash).
  Override in `.env` with `GEMINI_MODEL`.
- **Cost guard**: queries above 500 MB (dry-run) are sent back to the model to narrow;
  BigQuery itself rejects anything above 1 GB billed.
- Google APIs are not available from some regions; `analyst-doctor` detects this and says so.
