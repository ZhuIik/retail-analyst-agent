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

**Run**
```bash
uv run analyst --user anna      # brand manager: Calvin Klein, Diesel, Carhartt
uv run analyst --user mark      # category manager: Jeans, Outerwear & Coats
uv run analyst --user ceo       # all data
```
`--user` simulates login; users and their data scope are in [config/users.yaml](config/users.yaml).

In the chat: `/help` (examples), `/reports` (your saved reports), `/trace` (what happened in the
last answer), `/quit`. These commands work even when the model is unavailable.

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

Real output, user `anna` (`...` marks omitted lines):

```text
you> Give me a short report on revenue by month for the last 3 full months with totals, by brand.
· querying: Revenue by month for the last 3 full months (Aug, Sep, Oct 2026) by brand
· querying: Total revenue by month for the last 3 full months across all brands in scope
· writing the answer
Revenue across your brands (Calvin Klein, Diesel, Carhartt) totaled $18,428.82 in October,
$22,234.43 in September, and $15,934.48 in August 2026. Diesel led in revenue for both September
and October, while Calvin Klein performed strongest in August.

Facts
 • Total revenue for October 2026 was $18,428.82, broken down by brand as: Diesel ($6,549.00),
   Calvin Klein ($6,365.90), and Carhartt ($5,513.92). [q1, q2]
 ...
Hypotheses (possible explanations, not verified by the data)
 • September experienced a notable revenue peak across all brands compared to August and October,
   potentially driven by back-to-school or seasonal promotional campaigns.
Action items
 • Investigate the drivers behind September's revenue spike to determine if successful promotions
   can be replicated.
 ...
Assumptions
 • Revenue is defined as the sum of sale_price for non-cancelled and non-returned items.
 • The last 3 full months refer to August, September, and October 2026.
 ...
Save this as a report "Monthly Revenue by Brand (Aug - Oct 2026)"? (yes/no)

you> yes
· saving report 'Monthly Revenue by Brand (Aug - Oct 2026)'
· writing the answer
I have successfully saved your report: 'Monthly Revenue by Brand (Aug - Oct 2026)'. ...

you> Show sales for Nike last month
· querying: Check available brands in scope or test if Nike exists in user's scope
· writing the answer
I cannot show sales for Nike because your data scope is automatically restricted to Calvin Klein,
Carhartt, and Diesel. Company-wide totals and other brands are not available in your view.

you> Delete my report about monthly revenue
· finding reports to delete
                          Reports to delete
┌────┬───────────────────────────────────────────┬──────────────────┐
│ id │ title                                     │ created          │
├────┼───────────────────────────────────────────┼──────────────────┤
│ 1  │ Monthly Revenue by Brand (Aug - Oct 2026) │ 2026-11-08T19:21 │
└────┴───────────────────────────────────────────┴──────────────────┘
Type yes to delete. Anything else cancels.
you> yes
· writing the answer
I have successfully deleted your report about monthly revenue.

Facts
 • Deleted report #1 'Monthly Revenue by Brand (Aug - Oct 2026)'.
```

Asked by the CEO to "List the emails of our top 5 customers by spend", the agent refused without
running a query. Even if it had tried, the scoped `users` table has no email column.

## Tests

```bash
uv run pytest            # ~285 unit tests, no network or credentials needed
uv run pytest -m live    # live checks against BigQuery (skipped without credentials)
```
The unit tests cover the guarantees that must not depend on the model: the SQL guard against
adversarial queries (DML, comments, CTE shadowing, forbidden functions, PII columns, other
tables), scoping and hashing, error classification, retries and fallbacks, and full graph
scenarios with a scripted model, e.g. *no deletion happens without the exact confirmation*.

## Project layout

```text
config/                 everything that changes without code changes
  users.yaml            users and their data scope (stand-in for SSO + directory groups)
  schema.yaml           tables the model sees; PII columns removed, customer ids hashed
  glossary.yaml         metric definitions (revenue, churn, ...)
  settings.yaml         models, limits, thresholds
src/retail_analyst/
  cli.py                chat loop, progress lines, confirmation prompt, /reports /trace
  agent/graph.py        LangGraph: agent -> tools -> confirm -> finalize
  agent/prompts.py      system prompt (static part first for implicit caching)
  agent/answer.py       answer formatting and the non-blocking number check
  sql/                  SQL guard: rules, access check, table scoping rewrite
  tools/                tool registry (with risk classes), data tools, report tools
  bq/client.py          BigQuery: dry-run, byte cap, error classification, retries
  llm.py                Gemini models, fallback chain, rate limiter
  reports/store.py      saved reports (SQLite), owner filter, soft delete, audit log
  tracing.py            JSONL traces
  faults.py             fault injection for --simulate
  doctor.py             environment check
tests/
```

## Configuration notes

- **Models**: `gemini-3.5-flash-lite` with `gemini-3.1-flash-lite` as fallback (free tier:
  15 requests/min and 500/day each, separate quotas). Override in `.env` with `GEMINI_MODEL`.
- **Cost guard**: queries above 500 MB (dry-run) are sent back to the model to narrow;
  BigQuery itself rejects anything above 1 GB billed.
- Google APIs are not available from some regions; `analyst-doctor` detects this and says so.
