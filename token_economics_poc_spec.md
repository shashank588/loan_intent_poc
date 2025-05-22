
# Token‑Economics PoC — Project Requirement (v2)

## 1. Goal  
Create a **Python PoC + Streamlit dashboard** that, for a single transcript, runs the loan‑evaluation prompt against five cloud LLMs, captures token‑usage economics via **LangChain**, and writes everything (including the raw JSON answers) into an Excel workbook that the Streamlit UI will visualise.

---

## 2. Models in Scope

| # | Provider | Model ID |
|---|----------|----------|
| 1 | OpenAI   | `o3` |
| 2 | OpenAI   | `4o` |
| 3 | OpenAI   | `o4-mini` |
| 4 | GroqCloud | `llama-3-70b-instruct` |
| 5 | GroqCloud | `deepseek-r1` |

*No local/llama‑cpp runtimes in this PoC.*

---

## 3. Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Core runtime | **Python 3.11** | |
| LLM orchestration & token counting | **LangChain v0.2+** — use `get_num_tokens()` for any provider that doesn’t return a `usage` object |
| Async HTTP | `httpx` (used by LangChain internally) |
| Data wrangling & Excel | `pandas` + `openpyxl` |
| Config | `yaml` (via `pydantic‑yaml` if you like) |
| Dashboard | **Streamlit 1.33+** |

---

## 4. Directory Layout

```
token-poc/
├── data/
│   └── rubber_industry_transcript.txt
├── prompts/
│   └── loan_eval_prompt.txt
├── runs/
│   └── 2025-05-18T20-45-12_summary.xlsx
├── driver.py          # CLI batch runner
├── app.py             # Streamlit dashboard
├── config.yaml.example
├── requirements.txt
└── README.md
```

---

## 5. Workflow (`driver.py`)

1. **Load** transcript and base prompt.  
2. Iterate over *each* model:  
   - Build a LangChain `ChatModel` (OpenAI or GroqCloud wrapper).  
   - Send prompt → receive completion.  
   - **Token counting**  
     ```python
     prompt_tokens     = model.get_num_tokens(full_prompt)
     completion_tokens = model.get_num_tokens(response.content)
     ```  
     *If provider returns a reliable `usage` object, trust that instead.*  
   - **Cost calculation**  
     ```
     cost = (tokens / 1_000) × price_per_1k
     ```  
   - **Assemble a row**  
     ```json
     {
       "model": "openai/o3",
       "input_tokens": 123,
       "output_tokens": 456,
       "total_tokens": 579,
       "input_cost_usd": 0.61,
       "output_cost_usd": 0.68,
       "total_cost_usd": 1.29,
       "json_response": "{…full JSON…}"
     }
     ```
3. Append each row to a `pandas.DataFrame`.  
4. **Excel output**  
   - Workbook name → timestamped under `./runs/`.  
   - Sheet name → same timestamp (future multi‑file runs append new sheet).

---

## 6. Cost Reference (`config.yaml.example`)

```yaml
prices:
  openai:
    o3:
      prompt: 0.005
      completion: 0.015
    4o:
      prompt: 0.010
      completion: 0.030
    o4-mini:
      prompt: 0.003
      completion: 0.009
  groqcloud:
    llama-3-70b-instruct:
      prompt: 0.0008
      completion: 0.0012
    deepseek-coder-33b-instruct:
      prompt: 0.0006
      completion: 0.0009
```

---

## 7. Streamlit (`app.py`)

*Minimal dashboard features*

1. **Load** latest `./runs/*_summary.xlsx` or let user upload.  
2. Sidebar filters:  
   - Choose run timestamp / file.  
   - Toggle columns (e.g., hide JSON).  
3. Main view:  
   - **DataFrame** (styled) with numeric columns.  
   - **Bar chart**: `total_cost_usd` by model.  
   - **JSON expander**: click a row to inspect full LLM output and the prompt.

```python
import streamlit as st, pandas as pd
df = pd.read_excel(selected_file)
st.dataframe(df)
st.bar_chart(df.set_index("model")["total_cost_usd"])
if st.session_state.get("row_click"):
    st.json(df.loc[idx, "json_response"])
```

---

## 8. Requirements Snapshot (`requirements.txt`)

```
langchain==0.2.*
openai>=1.14.1
httpx>=0.27
pandas>=2.2
openpyxl>=3.1
pyyaml>=6.0
streamlit>=1.33
pydantic-yaml>=1.2
```

---

## 9. Error Handling

* Retry up to 3× on 429 / timeouts (exponential back‑off).  
* If a call fails, write `"status":"error"` in the row and keep going.  
* If JSON schema check fails, still store the raw text in `json_response` and flag a `json_valid: false` column.

---

## 10. Deliverables

| File | Purpose |
|------|---------|
| `driver.py` | Async runner producing the Excel workbook |
| `app.py` | Streamlit comparison UI |
| `config.yaml.example` | API keys & prices |
| `requirements.txt` | Exact dependency versions |
| `README.md` | Setup instructions & screenshots |

---

*Hand this file to Cursor IDE; once the scaffold is generated, refine the code and polish the Streamlit UX as needed.*
