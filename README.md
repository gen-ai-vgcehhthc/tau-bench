# TAU-Bench：三種 Agent 框架的比較

使用 [TAU-Bench](https://github.com/sierra-research/tau-bench)（Sierra Research）對三種主流 AI Agent 框架進行基準測試，模擬零售客服場景，量化各框架的**任務成功率**、**Token 消耗**與**執行時間**。

| 框架 | 說明 |
|------|------|
| **LangGraph** | LangChain 的圖狀 agent 框架，使用 `ChatOpenAI.bind_tools()` 手動 tool-calling 迴圈 |
| **MAF** (Microsoft Agent Framework) | 微軟新一代 agent 框架（前身 AutoGen），使用 `OpenAIChatClient` + 宣告式 `FunctionTool` |
| **CrewAI** | 多 agent 協作框架，底層透過 `litellm` 進行 tool-calling |

---

## 評測結果

**設定**：retail domain｜15 tasks（task 0–14）｜agent 與 user simulator 皆使用 `gpt-4o-mini`｜temperature 0

| 框架 | 成功率 | 平均 Tokens/task | 平均時間/task | 總 Tokens |
|------|:------:|:----------------:|:-------------:|:---------:|
| **MAF** | **7/15 (47%)** | 65,909 | 53.0s | 988,636 |
| **LangGraph** | 6/15 (40%) | 89,292 | 57.8s | 1,339,376 |
| **CrewAI** | 6/15 (40%) | 64,120 | **38.3s** | 961,795 |

### 各任務成功與否

| Task | LangGraph | MAF | CrewAI |
|:----:|:---------:|:---:|:------:|
| 0  | ❌ | ❌ | ❌ |
| 1  | ✅ | ❌ | ❌ |
| 2  | ✅ | ❌ | ❌ |
| 3  | ❌ | ❌ | ❌ |
| 4  | ❌ | ✅ | ❌ |
| 5  | ❌ | ❌ | ❌ |
| 6  | ✅ | ✅ | ❌ |
| 7  | ✅ | ❌ | ✅ |
| 8  | ❌ | ✅ | ✅ |
| 9  | ❌ | ✅ | ✅ |
| 10 | ✅ | ✅ | ✅ |
| 11 | ❌ | ✅ | ✅ |
| 12 | ❌ | ✅ | ❌ |
| 13 | ❌ | ❌ | ✅ |
| 14 | ✅ | ❌ | ❌ |

### 觀察

- **成功率**：三者接近（40–47%），MAF 略勝。`gpt-4o-mini` 在 retail 任務本就偏難（官方 `gpt-4o` 約 60%），失敗多源於模型推理能力，而非框架本身。
- **Token 效率**：CrewAI（64K）與 MAF（66K）相近；LangGraph 最高（89K），因為它在失敗任務上常反覆重試（如 task 5 用了 166K tokens）。
- **速度**：CrewAI 最快（38s），因為直接呼叫 `litellm`、沒有額外框架抽象層的開銷。
- **成功的任務各框架不完全重疊**，顯示框架的 prompt 組織與工具呼叫策略會影響不同類型任務的表現。

### 費用

三框架 agent 端合計約 3.29M tokens，加上 user simulator，本次評測總費用約 **$0.6–0.7 USD**。
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

### 為什麼採用獨立子專案（separate `pyproject.toml`）？

`agent-framework`（MAF）與 `crewai` 對 `opentelemetry-api` 的版本需求互相衝突
（MAF 需 `>=1.39.0`，CrewAI 需 `>=1.34.0,<1.35`），無法安裝於同一環境，
因此每個框架各自擁有獨立的 `uv` 虛擬環境與依賴。

---

## 運作原理

三個框架都實作 tau-bench 的 `Agent` 基底類別，核心是 `solve(env, task_index, max_num_steps)`：

1. `env.reset(task_index)` → 取得第一句使用者訊息
2. 進入迴圈：呼叫 LLM →
   - 若 LLM 回傳 **tool call** → 透過 `env.step(Action(name, kwargs))` 執行工具，將結果回填對話
   - 若 LLM 回傳 **純文字** → 透過 `env.step(Action("respond", {"content": ...}))` 傳給使用者，user simulator（`gpt-4o-mini`）生成回覆
3. 當 `env.step()` 回傳 `done=True` 時結束，由 tau-bench 自動評分（reward 0 或 1）

每個框架的差異在於「如何呼叫 LLM 與解析 tool call」：

- **LangGraph**：`bind_tools(parallel_tool_calls=False)`，每輪只取第一個 tool call。
- **MAF**：`FunctionTool` 採**宣告式**（不提供 `func`），避免 MAF 自動執行工具導致重複 API 呼叫；由我們手動路由到 `env.step()`。
- **CrewAI**：因 Crew/Task 抽象不支援多輪互動對話，改用其底層 `litellm.completion()` 直接控制 tool-calling 迴圈。

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

一鍵執行全部三個框架（依序，避免撞到 OpenAI 的 rate limit）：

```bash
./run_all.sh --model gpt-4o-mini --user-model gpt-4o-mini \
    --user-model-provider openai --start-index 0 --end-index 15
```

或單獨執行某個框架：

```bash
cd maf
uv run python run_benchmark.py --model gpt-4o-mini \
    --user-model gpt-4o-mini --user-model-provider openai \
    --task-ids 0 1 2 --log-dir ../results
```

執行時會顯示 **tqdm 進度條**，即時呈現完成進度與當前通過數，每完成一個任務也會印出一行結果：

```
maf:  47%|████▋     | 7/15 [06:11<07:04, 53.0s/task, passed=4/7]
Task   6 [PASS] reward=1.00 tokens=72517 time=44.7s
```

### 主要參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--env` | 評測領域（`retail` / `airline`） | `retail` |
| `--model` | agent 使用的模型 | `gpt-4o` |
| `--user-model` | user simulator 使用的模型 | `gpt-4o` |
| `--start-index` / `--end-index` | 任務範圍 | `0` / `-1`（全部）|
| `--task-ids` | 指定特定任務 ID | 無 |
| `--log-dir` | 結果輸出目錄 | `../results` |

> **注意**：三個框架同時並行執行會超過 OpenAI 的 TPM（tokens per minute）限制，
> 因此 `run_all.sh` 採**依序執行**。

---

## 輸出格式

每次評測會在 `results/` 產生一個 JSON，包含：

- `metrics`：每個任務的 reward、prompt/completion/total tokens、wall-clock time、步數
- `results`：完整對話軌跡（trajectory），可用於除錯與分析
- `summary`：整體成功率、平均 token、平均時間
