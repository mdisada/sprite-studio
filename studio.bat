@echo off
rem Sprite Studio - double-click me, or run from a terminal.
cd /d "%~dp0"
echo Starting Sprite Studio at http://127.0.0.1:8899 ...
python -m sprite_pipeline.ui --open
pause
