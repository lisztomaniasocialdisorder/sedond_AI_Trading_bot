from pathlib import Path

from binance_futures_harvester import BinanceFuturesHarvester

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    harvester = BinanceFuturesHarvester(
        symbol="ADAUSDT",
        project_root=PROJECT_ROOT,
        prometheus_port=9101,   # http://localhost:9101/metrics
    )
    harvester.run_forever()
