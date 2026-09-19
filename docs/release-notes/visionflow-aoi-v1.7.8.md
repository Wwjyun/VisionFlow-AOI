# VisionFlow AOI v1.7.8

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.6.3 起同一個 DLL，未重新編譯）。

檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更，CUDA source／header／ABI 未修改。本版修正相機機台第七次回報的 `060609`：無論在 CamExpert 手動把 Trigger Mode 切成 Off 或 On，都回報同一個碼。另外依現場回報，相機線速率最低 300 Hz。

## 依 Linea 官方手冊修正相機 Trigger Mode

Linea Camera Link 使用手冊（03-032-20206）寫明：

- `Trigger Selector`、`Trigger Source` 是**唯讀**，只有 `Trigger Mode` 可寫：Off＝內部 free-run，On＝外部 CC1。
- `AcquisitionLineRate` 只在 Trigger Mode Off 時可用。
- 線週期必須大於曝光 + 1 µs。

前幾版要先寫成功 selector，才算 Trigger Mode 設定成功；Linea 的 selector 是唯讀，所以每次都判定失敗。本版改為：

- 直接寫 Trigger Mode，並以**讀回值**判斷成功。已是 Off 但寫入被拒時不算失敗。
- 讀回與要求不符時回報 `E-0609`（連續模式要 Off、外部觸發要 On）；無法寫入也無法讀回時回報新碼 `E-0610`。
- 外部觸發（米輪）路徑原本也因同樣原因永遠切不到 On，本版一併修正。
- 寫線速率前會讀取相機回報的範圍，超出時自動調整到範圍內，例如 30 → 300 Hz，並在報告註明；Recipe 的值不變。
- 曝光寫入失敗時，`E-0602` 會附上這個線速率下的曝光上限，例如 5000 Hz 時約 199 µs。

## 新增：一行「讀回值」，一趟就能帶回完整狀態

數字短碼下方多一行英數字，**一併抄回**：

```text
TM=Off LR=300 LRMIN=300 LRMAX=48000 BLR=300 EXP=100 GAIN=1 W=16384 H=720 IMG=16384x720 MEAN=87.4
```

內容依序是：相機 Trigger Mode、線速率與相機回報的範圍、板卡線觸發頻率、曝光、增益、buffer 寬高、S7 實際收到的影像尺寸與平均灰階。GUI 診斷面板、打包版診斷對話框、CLI 與報告檔都有這一行。

## 現場操作

1. CCD 頁設定：**線速率 ≥ 300 Hz**，且**曝光 < 1,000,000 ÷ 線速率 − 1 µs**（例如 300 Hz 時曝光要小於約 3332 µs）。
2. 相機按「斷線」（米輪可保持連線）。
3. 「Sapera 診斷」→「執行相機診斷」，把**數字短碼**與**讀回值**兩行抄回。

## 相容性與限制

- 真實 Linea Mono 16K 的 Trigger Mode 讀回、S6／S7 與外部觸發仍待相機機台確認。
- 讀取線速率範圍用的 `SapFeature.GetValueMin/Max` 是選用功能；機台的 Sapera 沒有這個函式時會照原值寫入，讀回值顯示 `LRMIN=?`。
- CUDA DLL 與 v1.7.7 相同，其他 GPU／無 GPU 電腦驗收仍待目標環境。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
