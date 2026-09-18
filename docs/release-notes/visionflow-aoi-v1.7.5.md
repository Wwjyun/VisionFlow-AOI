# VisionFlow AOI v1.7.5

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.6.3 起同一個 DLL，未重新編譯）。

檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更，CUDA source／header／ABI 未修改。本版修正相機機台第四次回報的 `050503`（S5 SapBuffer 建立失敗）。這次回報已經證實 server、CCF 選擇正確，`SapAcquisition` 也建立成功。

## 修正：S5 建立 SapBuffer 失敗（`050503`）

真實 Sapera 的 `SapBufferWithTrash`／`SapBuffer` 建構子，來源參數型別是基底類別 **`SapXferNode`**，不是 `SapAcquisition`。參考程式寫 `new SapBufferWithTrash(2, _acquisition, …)` 能編譯，是因為 C# 會隱含轉換成基底類別。v1.7.1 起的程式只接受型別名稱剛好是 `SapAcquisition` 的建構子，所以兩個 buffer 類別都選不到，每次都在 buffer 失敗。

- 建構子挑選改為接受 `SapAcquisition` 與它的所有基底類別（以 .NET 反射取得，排除 `System.Object`），和 C# 編譯器的規則一致。
- 以舊版挑選函式重跑真實形狀 `(Int32, SapXferNode, SapBuffer+MemoryType)`，結果是選不到；修正後選到 `SapBufferWithTrash`（可回報 trash frame）。

## 新增：`E-0506` 找不到可用的 SapBuffer 建構子

若這台機器的 Sapera 仍提供其他形狀的建構子，S3（API 自檢）會直接回報 **`030506`**，不碰硬體，報告檔會列出機台實際提供的建構子。`E-0503` 現在只代表 buffer `Create()` 失敗（記憶體或 CCF 尺寸問題），兩種原因抄回的數字不同。S3 通過時，報告會寫出選到的 buffer 類別。

## 現場操作

1. CCD 頁先按相機的「斷線」（米輪可保持連線）。
2. 「Sapera 診斷」→「執行相機診斷」，把最上面那一列數字短碼抄回。
3. 預期 S5 會通過；接下來的 S6（參數寫入讀回）與 S7（Snap 一張，寬度應為 16384）是第一次在真實硬體上執行。

## 相容性與限制

- 真實 Xtium-CL MX4／Linea Mono 16K 的 S5–S8 仍待相機機台確認。
- CUDA DLL 與 v1.7.4 相同，其他 GPU／無 GPU 電腦驗收仍待目標環境。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
