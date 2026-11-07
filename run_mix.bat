@echo off
chcp 65001 >nul
echo ========================================================
echo       Automated Audio Pipeline - Running Mix...
echo ========================================================
python audio_mixer.py --config tracks.json
echo.
echo Mixing complete! Check the files in root and output/:
echo  - master_audio*.mp3
echo  - master_audio*.wav
echo ========================================================
pause

