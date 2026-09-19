# ARTIFACTS.md — 未追蹤產物地圖

本檔說明 repository 內**不進版控**的使用者產物放在哪裡、為什麼這樣分，以及搬動時的規則。這些檔案依 `AGENT.md` 一律不得 stage、不得刪除。整理日期：2026-09-14（2026-09-18 增列 `eiptest/` 與不變規則 7）。

## 目錄地圖

```
簡報/                                  投影片與其側錄檔（全部不進版控）
  interview/                           面試／作品集簡報
    VisionFlow_AOI_Interview_220W.pptx              作品集主檔
    VisionFlow_AOI_Interview_220W_v2_含架構五圖.pptx  加入五張架構圖的版本
    VisionFlow_AOI_面試_系統架構五張圖譜.pptx         純架構圖版本
    *.pptx.inspect.ndjson                            每份簡報的逐頁側錄（與簡報同名相鄰）
    slides/                            由 VisionFlow_AOI_Interview_220W.pptx 匯出的 slide-*.png
  phase2/                              第二階段改動說明簡報
    VisionFlow_AOI_Phase2_架構與兩項重大改動說明.pptx（＋ _v8、_v9 版本）
    Phase2_改動一_PLC逐排訊號.pptx       ＋ .inspect.ndjson
    Phase2_改動二_GPU加速.pptx           ＋ .inspect.ndjson

架構圖/                                 圖譜產物（Archify，全部不進版控）
  01_main_architecture … 09_data_movement_current
                                       現行 9 張圖，每張一組 .json（規格）＋ .html（互動圖）＋ .png（靜態圖）
                                       `06`–`09` 另附 `.visual-check.*` 截圖與收據
  README.md                            9 張圖的閱讀指南與事實更正說明
  runtime-architecture/                另一組單圖產物（原本散在 repo 根目錄）
                                       .archify.json 規格、.html、.png 截圖、.visual-check.json 收據
  charts/                              5 份 HTML 圖表（01_overview … 05_dashboard）
  _backup_20260911/                    2026-09-11 的舊版 01–05 圖譜快照（凍結，不再更新）

cuda_course/                           30 天 CUDA 自學課程（Markdown，含練習題說明）
interview_prep/                        面試準備教材 L00–L05＋README
xx_ccd/                                線掃相機擷取 C# WinForms 專案（Sapera LT＋LSI-8181 米輪）
                                       另一個 repository 的複本（自帶 `.git`，見不變規則 7）
                                       只作 Todo.md P11 移植的行為參考，不帶進產線
eiptest/                               Keyence PLC 通訊工具（KV-8000／KV-XLE02；上位鏈結 TCP 8501＋EtherNet/IP CIP）
                                       另一個 repository 的複本（remote `keyence-plc-hud`，自帶 `.git`，見不變規則 7）
                                       內含 Python `src/`、Electron `electron/`＋`frontend/`，以及 `env/`、`node_modules/` 本機依賴
                                       用途：PLC 通訊層與逐排訊號的參考實作，對應 `docs/reports/Phase2_改動一_PLC逐排訊號評估.md`
.codex_ppt_build/                      產生 `簡報/` 的 Node 工作區
  interview_deck.mjs / interview_deck_v2.mjs / arch_interview_deck.mjs / vp_decision_deck.mjs
                                       deck 產生腳本，輸出路徑已指向 `簡報/`
  assets/                              腳本內嵌用的架構圖 PNG（架構圖/01–09 的複本）
  renders*/                            各版本的逐頁 PNG 與 layout.json（產物，可刪可重建）
docs/reports/*.md                      尚未進版控的技術／階段報告草稿
```

## 不變規則

1. **`架構圖/01`–`09` 的路徑不可再搬。** 這九組被 20 多處引用：9 個 archify 規格檔的 `meta.output`、`06`–`09` 的 `.visual-check.json` 絕對路徑、`docs/reports/Phase2_*評估.md` 的圖片連結、以及 4 個 deck 腳本內文。要動就得同一輪把所有引用一起改。
2. **`.inspect.ndjson` 必須與同名 `.pptx` 放在一起。** 它是側錄檔，分開會失去對應。
3. **只整組搬移，不拆散。** 要換位置就搬整個 `簡報/interview/` 這種層級，並在同一次變更內更新所有引用。
4. **`_backup_20260911/` 是凍結快照**，內容不再更新，其內部規格檔的 `output` 欄位仍指向舊路徑，屬預期現象。
5. **`.codex_ppt_build/assets/` 是複本**，來源是 `架構圖/01`–`09` 的 PNG。重跑 deck 腳本前若要更新圖，先更新 `架構圖/`，再同步複製到 `assets/`。
6. **重跑 deck 腳本會覆寫 `簡報/` 內的 pptx**，不會再產生根目錄檔案（4 個腳本的 `OUT` 已一併修正）。
7. **`xx_ccd/` 與 `eiptest/` 是巢狀 repository，永遠不可 stage。** 兩者各自帶 `.git`，所以母 repo 的 `git status` 只會把它們各顯示成一行未追蹤項目（`?? xx_ccd/`、`?? eiptest/`），內部檔案不會展開；`git add -A`／`git add .` 會把它們收成 gitlink（子模組指標），既污染 VisionFlow 又會誤導另一個 repository 的歷史。要 stage 一律明列檔案路徑（見 `AGENT.md` 的 Git 規則）。巢狀 repo 內部若有未提交變更，須進該目錄自行 `git status` 才看得到。

## 已知狀況

- `架構圖/runtime-architecture/runtime-architecture.visual-check.json` 記錄的是 **`"ok": false` / `"status": "fail"`** 的視覺檢查結果，尚未重跑成功。
- `docs/reports/架構圖_升級檢視報告.md` 是 2026-09 的檢視報告，內容對應當時的 5 張圖狀態，**未隨本次整理改寫**（報告保留當時事實）。
- 本目錄下的產物全部未追蹤，`git status` 會持續顯示它們；這是刻意設計（見 `AGENT.md`），不是遺漏。
