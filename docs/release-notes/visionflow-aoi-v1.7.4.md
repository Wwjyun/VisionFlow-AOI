# VisionFlow AOI v1.7.4

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.6.3 起同一個 DLL，未重新編譯）。

本版沒有 v1.7.3。檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更，CUDA source／header／ABI 未修改。修正對象是相機機台第三次回報：**米輪已連上**，但相機診斷回報 `069998`（第 6 步失敗、沒有錯誤碼）。

## 修正：診斷失敗一定帶錯誤碼，不再出現 `9998`

S4、S6、S7 在 Sapera 丟出例外時，畫面短碼只寫中文原因、沒有 `E-xxxx`，所以數字短碼只能顯示 `9998`，現場抄回來也無法判斷原因。現在每一個 FAIL 短碼都以錯誤碼開頭，例如：

```text
S6 FAIL E-0502 SapAcquisition 建立失敗：…
數字短碼：060502
```

沒有分類的例外也會顯示 `E-0901`（`060901`），不再是 `9998`。

## 修正：相機已連線時不能執行相機診斷

診斷會自己在同一張 Xtium 擷取卡上建立 `SapAcquisition`。如果 CCD 頁的相機已經連線，擷取卡已被占用，S5／S6 會失敗，看起來卻像硬體故障。現在相機連線時按「執行相機診斷」，畫面會以提示拒絕，並提醒**先按「斷線」再診斷**。米輪不受影響，可以保持連線。

## 修正：S5 不再把全部失敗的物件報成 PASS

- 沒有選擇 server 時，S5 直接回報新錯誤碼 **`E-0404` 尚未選擇 Sapera 擷取卡（server）**（`050404`），不會碰硬體。連線時沒有 server 也改成 `E-0404`。
- `SapAcquisition`、buffer 或 `SapAcqToBuf` 任一個建立失敗時，S5 就是 FAIL，並帶第一個錯誤碼（例如 `050502`）。以前 S5 會顯示 PASS，錯誤要到 S6 才出現。`SapAcqDevice` 失敗仍只記在報告檔，相機 feature 寫入由 S6 回報。

## 修正：打包版 `--sapera-diagnose` 讀取機台設定檔

`VisionFlow AOI.exe --sapera-diagnose` 與 `main.py --sapera-diagnose` 以前傳入空白的 server／CCF，所以 S6 不可能連上。現在這兩個入口與 CCD 頁一樣，使用 `config\ccd_machine.json` 儲存的位置。報告開頭會寫出實際讀取的設定檔路徑，以及診斷使用的 server／CCF。

## 現場操作

1. CCD 頁先按相機的「**斷線**」（米輪可保持連線）。
2. 「Sapera 診斷」→「執行相機診斷」，把最上面那一列**數字短碼**抄回。
3. 若 S5 或 S6 是 `0502`，代表 `SapAcquisition` 建立失敗。請確認 CamExpert 等其他取像程式已關閉，且 CCF 對應 Xtium-CL MX4。

## 相容性與限制

- Sapera LT 相機仍未完成實機驗收。本版讓失敗原因能以數字短碼帶回，並排除診斷與已連線相機搶用擷取卡的狀況，但真實 Xtium-CL MX4／Linea Mono 16K／CCF 仍需在相機機台確認。
- 米輪已在相機機台連線成功（使用者回報）；自動連線、Compare 寫入與外部觸發等完整驗收仍待實測。
- CUDA DLL 與 v1.7.2 相同，其他 GPU／無 GPU 電腦驗收仍待目標環境。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
