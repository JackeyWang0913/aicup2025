# AICUP 任務流程說明

本文件說明您在 AICUP 2025 中，針對 Task 1（語音轉文字）與 Task 2（PHI 擷取）的整體流程與架構設計，並指出各階段所使用的模型、方法與資料處理步驟。

---

## Task 1：語音辨識與時間戳對齊

### 1. 使用模型與工具

- **模型架構**：WhisperX（基於 OpenAI Whisper 模型）
- **語音辨識模型等級**：`medium`（比 base 精準，使用 float16 計算降低 GPU 占用）
- **對齊模型來源**：WhisperX 內建的 alignment 模組

### 2. 資料處理步驟

1. **音訊前處理**：
   - 使用 `librosa` 將 `.wav` 檔載入為 16kHz 標準格式的 numpy array。
2. **音訊暫存**：
   - 將音訊寫入暫存 `.wav` 檔案供 WhisperX 使用。
3. **語音轉文字（ASR）**：
   - 使用 WhisperX 進行語音辨識並指定語言為英文 `language="en"`。
4. **時間戳對齊**：
   - 使用 WhisperX 對齊模組輸出每個 word 的 `start`, `end` 時間戳。
5. **結果輸出**：
   - 輸出至 `task1_answer.txt`（逐句結果）與 `task1_answer_timestamps.json`（逐詞時間戳）。

### 3. 優化設計

- 關閉非必要警告與 PyTorch Lightning 訊息，提升輸出清晰度。
- 預先指定語言避免每筆重新偵測語言造成延遲。

---

## Task 2：PHI 擷取與時間標註

### 1. 使用模型與訓練設定

- **語言模型**：DeepSeek LLM 7B Chat 版本
- **量化方式**：BitsAndBytes 4-bit 量化（nf4 + double quant）
- **微調方式**：LoRA（對 `q_proj`, `k_proj`, `v_proj`, `o_proj` 插入低秩適應）
- **特殊 Token 設計**：
  - 使用 `<|endoftext|>`, `<|END|>`, `<|pad|>`, `\n\n####\n\n` 作為 BOS, EOS, PAD, SEP tokens

### 2. 資料格式設計

- 訓練輸入格式：
  ```
  <|endoftext|> 語音辨識轉錄內容

  ####

  PHI1: 類別1
  PHI2: 類別2
  ... <|END|>
  ```
- 使用 template 產生完整 prompt，再與標註一起對應為 Seq2Seq 格式進行訓練。

### 3. 推論與後處理流程

1. **產生 prompt 並送入模型**：
   - 使用訓練時的 template 將語音轉錄餵入 LLM。
2. **模型生成**：
   - 利用 `.generate()` 生成模型輸出。
3. **結果處理**：
   - 從 `SEP` 後方抽取 PHI 結果段落。
   - 搭配 Task 1 的 `timestamps`，比對生成的文字在原始音訊中出現的位置。
   - 匹配後輸出標註格式（含時間戳）。

### 4. 輸出格式

- 最終輸出為符合 Task2 評分格式的：
  ```
  <fid>\t<phi類型>\t<start>\t<end>\t<entity>
  ```

---

## 整體流程總結

```text
.wav 音檔 → WhisperX 語音辨識 → 時間戳對齊 →
      ↓                          ↑
     文字 ----------------→ DeepSeek LLM → PHI 抽取與時間對齊
```

---

## 可視化建議（報告可放入圖片）

1. WhisperX 處理流程圖（語音 → 文本 → 對齊）
2. Task 1 + Task 2 整合流程圖（上下游資料如何串接）
3. DeepSeek 微調架構示意圖（顯示 LoRA 參數插入位置）
4. 損失折線圖（已於程式碼中產生 loss\_curve.png）

