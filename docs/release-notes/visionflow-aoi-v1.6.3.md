# VisionFlow AOI v1.6.3

這是 Windows x64 CUDA-enabled 更新，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（SHA-256 `38433800568FAB7BBD8E7007A970ADE20B11B2960FEDD319829C78B8167B2345`）。

## CCD 線掃相機與相機直連監控

- 新增「CCD 控制」畫面：相機與 LSI-8181 米輪連線、取像參數、觸發模式、存圖與 CMP 設定；缺相機 binding、`LSI8181_64.dll` 或硬體時主程式照常啟動，畫面顯示原因。Sapera LT 相機 binding 尚未完成，目前可用模擬相機驗證流程。
- 產品層相機參數存入 Recipe 選用 `camera` 區段；機台層設定存本機設定檔。Recipe 設計的相機區塊僅管理模式可編輯；OP 不可進入 CCD 控制。
- 監控模式可選「監控資料夾」或「相機直連」。相機直連以記憶體交接檢測觸發完成的 frame，結果格式與資料夾監控相同。
- 相機直連會一邊檢測、一邊把每張 frame 存到本次監控資料夾的 `raw/`（CCD 機台存圖格式、`.tmp` 原子寫入），不需先寫檔再讀回；「開啟原始影像」直接開啟該檔。16384×13000 單通道 frame 實測寫 BMP 774 ms、讀回 556 ms，此流程省下讀回並讓寫檔與檢測重疊。

## GPU 與效能

- `GpuExecutionSession` 重用 host 影像緩衝並註冊為 pinned memory（新增 optional `vf_host_register_u8`／`vf_host_unregister_u8`）；同一未變更檔案再檢測時重用解碼結果。
- GUI 單張、預熱、批量、資料夾與相機監控共用同一個 GPU session；Designer 儲存 Detector 參數不再重建 CUDA session。載入 GPU Recipe 與影像後自動背景預熱，監控與批量開始前檢查 CUDA context。
- CUDA context 顯存改為逐類統計（新增 optional `vf_context_memory_stats_v1`），舊 DLL 自動退回總量口徑；ABI v1 不變。
- RTX 3090 正式基準（16384×13000 BMP、6 個 12000×2000 ROI、`202-CS-SN-1`，CPU／GPU 交錯 5 輪）：CPU 5392.9 ms、GPU **271.5 ms**（median，**19.86×**），5/5 輪判定欄位相同；連續 100 張 warm median／P95 226.8／232.1 ms。

## GUI

- 「檢測結果」新增效能分析面板，顯示各階段耗時、Detector 子階段、device/host split、H2D／D2H 與 VRAM 狀態，資料取自本次執行結果。
- 工程／管理模式新增 CPU／GPU 對照執行，比較判定欄位與各階段倍數，不修改 Recipe。
- 大圖預覽改用降採樣金字塔：只把最粗層轉成畫面，放大時才轉換可見區域。16384×13000 影像 GUI 行程穩定記憶體 1503 → 904 MB，GUI 執行緒顯示 190～270 → 7～12 ms。
- 1000 張批量完成時的表格填入 730 → 54 ms，不再卡住畫面；worker 結果以 Python 物件跨執行緒傳遞，避免大批量摘要轉換造成凍結。
- 執行進度訊息改為繁體中文；Recipe 設計的 Detector 清單加寬，每列 GPU 開關可見。

## 輸出

- Matrix CSV 的 NG 格改寫該 Tile 的缺陷類型（去重，以 `; ` 分隔），PASS 格維持空白；`matrix_summary` 工具彙總結果不變。
- 新增 `999-FLOW-TEST` 流程驗證 Detector 與 Recipe。

## 相容性與限制

- CUDA DLL 以 RTX 3090／compute capability 8.6 驗證；其他 GPU 必須支援包內 `sm_86` binary。沒有相容 NVIDIA GPU 時可選「僅 CPU」或「GPU 優先，失敗改用 CPU」。
- Sapera LT 相機 binding 尚未實作；CCD 相機、米輪與相機直連監控尚未在相機機台實機驗證，本版以模擬裝置與自動測試驗證流程。
- 效能數據來自固定 seed 合成影像；真實產線影像與判定門檻仍需在目標產線驗收。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
