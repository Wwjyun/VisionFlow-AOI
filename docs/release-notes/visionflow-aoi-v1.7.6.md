# VisionFlow AOI v1.7.6

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.6.3 起同一個 DLL，未重新編譯）。

檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更，CUDA source／header／ABI 未修改。本版修正相機機台第五次回報的 `060601`：S5 已通過（v1.7.5 的 buffer 修正有效），相機也已連線，但相機線速率寫入被拒。

## 修正：相機線速率（`AcquisitionLineRate`）寫入方式與參考程式一致

參考程式依序嘗試 Int64、小數字串（例如 `30.00`）、整數字串（`30`）三種寫法，前一版只試了 Int64。GenICam 的 `AcquisitionLineRate` 是浮點數 feature，可能只接受字串。本版依相同順序嘗試，三種都失敗才回報 `E-0601`；報告檔會列出寫入前的值與存取模式。若仍是 `E-0601`，最可能是要求值低於相機最低線速率，請用 CamExpert 確認可寫範圍。

## 改進：一次診斷看到更多結果

- **S6 部分參數寫入失敗時，S7 仍會 Snap 一張**，S8 仍會清理。S6 沒有連上時，兩者才 SKIP。
- **數字短碼會列出同一步的所有錯誤碼**：例如 `060601 060602` 代表 S6 的線速率與曝光都寫入失敗。整列照抄即可，組數不一定是 8 組。
- 板卡端內部線觸發失敗改用新碼 **`E-0608`**，不再與相機端的 `E-0601` 共用。

## 現場操作

1. CCD 頁先按相機的「斷線」（米輪可保持連線）。
2. 「Sapera 診斷」→「執行相機診斷」，把最上面那一列數字短碼**整列**抄回。
3. S7 通過時會顯示影像尺寸，寬度應為 16384；若有 S7 那一組數字，也請一起抄回。

## 相容性與限制

- 真實 Linea Mono 16K 的線速率可寫範圍與 S7 取像仍待相機機台確認。
- CUDA DLL 與 v1.7.5 相同，其他 GPU／無 GPU 電腦驗收仍待目標環境。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
