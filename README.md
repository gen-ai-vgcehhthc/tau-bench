# TAU-Bench：三種 Agent 框架的比較

使用 [TAU-Bench](https://github.com/sierra-research/tau-bench)（Sierra Research）對三種主流 AI Agent 框架進行基準測試，模擬零售客服場景，量化各框架的**任務成功率**、**Token 消耗**與**執行時間**。

| 框架 | 說明 |
|------|------|
| **LangGraph** | LangChain 的圖狀 agent 框架，使用 `ChatOpenAI.bind_tools()` 手動 tool-calling 迴圈 |
| **MAF** (Microsoft Agent Framework) | 微軟新一代 agent 框架（前身 AutoGen），使用 `OpenAIChatClient` + 宣告式 `FunctionTool` |
| **CrewAI** | 多 agent 協作框架，底層透過 `litellm` 進行 tool-calling |

---

## 評測結果

**設定**：retail domain｜15 tasks（task 0–14）｜**每個任務跑 3 trials**｜agent 與 user simulator 皆使用 `gpt-4o-mini`｜temperature 0

| 框架 | 成功率 (run) | 95% CI | pass³ | pass-any | Tokens/run | 時間/run |
|------|:-----------:|:------:|:-----:|:--------:|:----------:|:--------:|
| **LangGraph** | 19/45 (42.2%) | [29.0%, 56.7%] | 4/15 | 9/15 | 79,027 | 58.1s |
| **MAF** | 18/45 (40.0%) | [27.0%, 54.5%] | 1/15 | 11/15 | 66,814 | 56.0s |
| **CrewAI** | 18/45 (40.0%) | [27.0%, 54.5%] | 2/15 | 10/15 | 78,372 | **49.0s** |

> `pass³` = 3 次 trials 全過的任務數（可靠度）；`pass-any` = 至少過 1 次的任務數。

### 各任務 3 trials 通過數（0–3，`~` 表示結果不一致 / flaky）

| Task | LangGraph | MAF | CrewAI |
|:----:|:---------:|:---:|:------:|
| 0  | 0 | 1 ~ | 2 ~ |
| 1  | 2 ~ | 1 ~ | 0 |
| 2  | 1 ~ | 0 | 0 |
| 3  | 0 | 0 | 0 |
| 4  | 0 | 1 ~ | 0 |
| 5  | 2 ~ | 1 ~ | 2 ~ |
| 6  | 1 ~ | 0 | 1 ~ |
| 7  | 0 | 2 ~ | 0 |
| 8  | 0 | 2 ~ | 1 ~ |
| 9  | 0 | 1 ~ | 1 ~ |
| 10 | 3 | 2 ~ | 3 |
| 11 | 3 | 2 ~ | 3 |
| 12 | 3 | 3 | 1 ~ |
| 13 | 3 | 2 ~ | 2 ~ |
| 14 | 1 ~ | 0 | 2 ~ |

---

## 框架是否影響結果？

**結論：以目前的設定，框架對「成功率」沒有可辨識的影響——差異完全被隨機性淹沒。**

證據：

1. **成功率三者幾乎相同**：42.2% / 40.0% / 40.0%，最大差距僅 **1 個 run（共 45）**。三者的 95% 信賴區間（皆約 27–57%）幾乎完全重疊，差異不具統計顯著性。

2. **隨機性主導結果**：**15 個任務中有 14 個是 flaky**——至少一個框架在 3 次 trials 給出不一致的結果（1 或 2 次通過）。也就是說，「同一框架、同一任務跑兩次」的變異，比「不同框架之間」的差異還大。主因是 user simulator 也是 `gpt-4o-mini`，每次生成的對話都不同。

3. **可靠度都很低**：能穩定 3 次全過的任務只有 1–4 個。單次評測得到的「某框架贏」其實是抽樣雜訊。

換句話說：**要可靠比較框架，單次 15-task 評測遠遠不夠**，需要更多 trials + 更多任務 + 統計檢定。本次 3 trials 的價值正是揭露了這個雜訊量級。

### 比較可靠的差異：成本與速度

成功率受雜訊影響大，但 **token 與時間是結構性差異，相對穩定**：

- **速度**：CrewAI 最快（49s/run），因為直接呼叫 `litellm`、沒有框架抽象層開銷。
- **Token**：MAF 最省（66.8K/run）；LangGraph（79K）與 CrewAI（78K）較高，前者源於失敗任務的反覆重試。

這些差異來自框架的實作架構，是即使在雜訊下也能觀察到的真實區別。

### 費用

三框架共 135 runs、約 10.1M tokens，加上 user simulator，本次評測總費用約 **$2–3 USD**。
（`gpt-4o-mini` 定價：input $0.15 / output $0.60 每 1M tokens）

---

## 專案結構

```
.
├── run_all.sh          # 一鍵依序執行三個框架
├── results/            # 評測結果 JSON（不納入版控）
├── langgraph/
│   ├── pyproject.toml
│   ├── run_benchmark.py
│   └── src/{agent.py, metrics.py}
├── maf/
│   ├── pyproject.toml
│   ├── run_benchmark.py
│   └── src/{agent.py, metrics.py}
└── crewai/
    ├── pyproject.toml
    ├── run_benchmark.py
    └── src/{agent.py, metrics.py}
```

---

## 執行方式

### 前置需求

- Python 3.13
- [`uv`](https://github.com/astral-sh/uv)
- OpenAI API key

### 設定 API key

在專案根目錄建立 `.env`：

```
OPENAI_API_KEY=sk-...
CREWAI_TRACING_ENABLED=false
```

各子專案以 symlink 共用此 `.env`。

### 安裝依賴

```bash
cd langgraph && uv sync && cd ..
cd maf && uv sync --prerelease=allow && cd ..
cd crewai && uv sync && cd ..
```

### 執行評測

一鍵執行全部三個框架（依序，避免撞到 OpenAI 的 rate limit）。
`--num-trials` 指定每個任務重複跑幾次，用來量化隨機性：

```bash
./run_all.sh --model gpt-4o-mini --user-model gpt-4o-mini \
    --user-model-provider openai --start-index 0 --end-index 15 --num-trials 3
```

或單獨執行某個框架：

```bash
cd maf
uv run python run_benchmark.py --model gpt-4o-mini \
    --user-model gpt-4o-mini --user-model-provider openai \
    --task-ids 0 1 2 --num-trials 3 --log-dir ../results
```

執行時會顯示 **tqdm 進度條**，即時呈現完成進度與當前通過數，每個 run 也會印出一行結果：

```
maf:  47%|████▋     | 21/45 [18:33<21:12, 53.0s/run, passed=8/21]
Task   6 trial 0 [PASS] reward=1.00 tokens=72517 time=44.7s
```

### 主要參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--env` | 評測領域（`retail` / `airline`） | `retail` |
| `--model` | agent 使用的模型 | `gpt-4o` |
| `--user-model` | user simulator 使用的模型 | `gpt-4o` |
| `--start-index` / `--end-index` | 任務範圍 | `0` / `-1`（全部）|
| `--task-ids` | 指定特定任務 ID | 無 |
| `--num-trials` | 每個任務重複次數 | `1` |
| `--log-dir` | 結果輸出目錄 | `../results` |

> **注意**：三個框架同時並行執行會超過 OpenAI 的 TPM（tokens per minute）限制，
> 因此 `run_all.sh` 採**依序執行**。

---

## 輸出格式

每次評測會在 `results/` 產生一個 JSON，包含：

- `metrics`：每個 run（task × trial）的 reward、prompt/completion/total tokens、wall-clock time、步數
- `results`：完整對話軌跡（trajectory），可用於除錯與分析
- `summary`：run 成功率、平均 per-task 通過率、pass^k 可靠度、平均 token / 時間
