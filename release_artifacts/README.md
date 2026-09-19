# 本機發行工件

此資料夾集中本機建立或保留的 Windows 發行 ZIP，包括 VisionFlow AOI、Utility Tools 與 Traditional CV Tuning Tool。

- ZIP 由 repository 的 `*.zip` 規則忽略，不得提交 Git。
- 正式發行檔保留既有版本化名稱，不覆寫同版本工件。
- `packaging/scripts/build_utility_tools.ps1` 會直接將合集 ZIP 寫入此資料夾。
- VisionFlow AOI 與 Traditional CV Tuning Tool 的 ZIP 也應在驗證完成後建立或移至此處。

GitHub Release 發布時，應從這裡選取已完成 smoke、大小與 SHA-256 驗證的版本化 ZIP。
