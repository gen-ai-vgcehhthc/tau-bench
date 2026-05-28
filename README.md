# TAU-Bench：三種 Agent 框架的比較

使用 [TAU-Bench](https://github.com/sierra-research/tau-bench)（Sierra Research）對三種主流 AI Agent 框架進行基準測試，模擬零售客服場景，量化各框架的**任務成功率**、**Token 消耗**與**執行時間**。

本專案做了**兩輪實驗**：

- **第一輪「對齊實作」**：三個框架都寫成同一種「單 agent + 手動 tool-calling 迴圈」，盡量公平。
- **第二輪「原生實作」**：每個框架改用其**官方推薦的原生寫法**，看框架差異是否會浮現。

| 框架 | 第一輪：對齊實作 | 第二輪：原生實作（官方推薦） |
|------|------|------|
| **LangGraph** | `ChatOpenAI.bind_tools()` 手動迴圈 | `create_react_agent`（prebuilt ReAct agent） |
| **MAF** (Microsoft Agent Framework) | `OpenAIChatClient` + 宣告式 `FunctionTool` | 原生 `Agent` + `agent.run()` 自動執行工具 |
| **CrewAI** | 底層直接呼叫 `litellm` | 原生 `Agent.kickoff()` + `BaseTool` |

> 第二輪的共用設計：把「跟使用者對話」也做成一個 `talk_to_user` 工具（內部走 `env.step` 的 respond action），讓每個框架的**原生 runner** 自動驅動整段多輪對話。

---

## 第一輪結果：對齊實作

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

## 第二輪結果：原生實作

第一輪三個框架被寫成同一種「最小公分母」，把框架的差異化特徵都抹平了。第二輪改用各框架**官方推薦的原生寫法**（見上方表格），重新檢驗框架差異是否會浮現。

**設定**：retail domain｜tasks 0–4（5 tasks × 1 trial）｜`gpt-4o-mini`｜temperature 0

| 框架（原生） | 成功率 | 平均 Tokens/task | 平均時間/task |
|------|:------:|:----------------:|:-------------:|
| **LangGraph** (`create_react_agent`) | 3/5 | 82,405 | 53.2s |
| **MAF** (`agent.run()`) | 0/5 | 75,995 | 42.8s |
| **CrewAI** (`Agent.kickoff()`) | 1/5 | **610,514** | 55.3s |

### 最大發現：CrewAI 原生寫法 token 暴量

把第二輪（原生）和第一輪（對齊）在**同樣 tasks 0–4** 的 token 對照：

| 框架 | 對齊實作 Tokens/run | 原生實作 Tokens/task | 變化 |
|------|:------------------:|:-------------------:|:----:|
| LangGraph | 84,546 | 82,405 | ≈ 持平 |
| MAF | 67,964 | 75,995 | ≈ 持平 |
| CrewAI | 97,513 | 610,514 | **↑ 約 6.3 倍** |

CrewAI 的單一任務 token 從 22 萬一路飆到 **181 萬**（per-task：223K / 326K / 141K / 550K / **1810K**）。原因是 **CrewAI 的 `Agent.kickoff()` 是無狀態的**：要做多輪對話，每一輪都得把整段歷史（含龐大的 retail policy backstory）重新餵進去，且每次 kickoff 內部又自帶 ReAct scaffolding 多跑幾次 LLM。**LangGraph 與 MAF 的原生 runner 原生支援有狀態的多輪迴圈，token 幾乎不變。**

這就是「換成原生寫法後浮現的框架差異」——但它出現在**成本/架構**維度，不是成功率。

### 成功率仍然分不出高下

原生實作的成功率（3/5、0/5、1/5）樣本太小（5 tasks × 1 trial），且我們在第一輪已證明成功率被模型能力與 user simulator 的隨機性主導。MAF 的 0/5 並非實作壞掉（驗證時 task 1 有 PASS），而是小樣本 + flaky 的正常波動。**換原生寫法沒有讓成功率分出差別，符合「框架是 orchestration layer、上限由模型決定」的預期。**

### 為什麼沒繼續用 gpt-4o 比較

原本想用更強的 gpt-4o 看差異是否浮現，但 CrewAI 原生 ~610K tokens/task 在 gpt-4o 下約 **$0.8/task**，$4 預算只夠 ~3 tasks，仍不足以壓過成功率的雜訊。既然**框架的真實差異（成本/架構）在 gpt-4o-mini 已經清楚浮現**，便不再投入更貴的 gpt-4o 評測。

### 兩輪總結

1. **成功率**：兩輪都無法分出框架高下——被模型能力與 user simulator 隨機性主導（這是 TAU-Bench 用 LLM 模擬使用者的結構性特徵）。
2. **成本/架構**：才是框架的真實差異。**原生寫法反而放大了它**——CrewAI 的 Agent 抽象不是為有狀態多輪對話設計的，硬套上去成本暴增 6 倍；LangGraph / MAF 的原生 agent runner 則原生支援，成本持平。
3. **方法論**：「框架理論上應該一樣」只在 prompt 逐字相同、模型與 user 都確定時成立；實務上框架的 prompt 包裝、訊息格式、狀態管理都不同，差異會經 user simulator 放大成完全不同的軌跡。

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
