@echo off
setlocal
cd /d C:\Users\brian\trading
python harvesters\csv_to_parquet.py %*
endlocal
