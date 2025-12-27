@echo off
:: 切換編碼為 UTF-8，解決中文亂碼問題
chcp 65001 > nul
title 導盲系統看門狗 (Watchdog)

:loop
cls
echo ==========================================
echo 正在啟動導盲系統...
echo ==========================================

:: 執行 Python 主程式
python main.py

:: 檢查程式是如何結束的
if %errorlevel% neq 0 (
    echo.
    echo [警告] 程式異常崩潰！(代碼: %errorlevel%)
    echo 正在發出警報並重啟...

    :: 呼叫 Windows 語音報警
    powershell -c "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('系統異常，正在重新啟動');"

    :: 等待 3 秒再重啟
    timeout /t 3
    goto loop
) else (
    echo.
    echo 程式已正常關閉。
    pause
    exit
)