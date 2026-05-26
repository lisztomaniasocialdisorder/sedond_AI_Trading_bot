@echo off
setlocal
cd /d C:\Users\brian\trading
python harvesters\db_to_parquet.py %*
endlocal
