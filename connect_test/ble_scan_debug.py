"""
周辺のBLEデバイスをすべてスキャンして,名前とアドレスを表示するデバッグ用スクリプト.
AccelSensorが見つからない場合,このスクリプトでまず何が見えているか確認する.

使い方:
    python ble_scan_debug.py
"""

import asyncio
from bleak import BleakScanner


async def main():
    print("周辺のBLEデバイスを5秒間スキャンします...")
    devices = await BleakScanner.discover(timeout=5.0)

    if not devices:
        print("BLEデバイスが1つも見つかりませんでした.")
        print("→ macOSのBluetooth権限(システム設定 > プライバシーとセキュリティ > Bluetooth)を確認してください.")
        return

    print(f"{len(devices)}台のデバイスが見つかりました:\n")
    for d in devices:
        name = d.name if d.name else "(名前なし)"
        print(f"  名前: {name:30s} アドレス: {d.address}")


if __name__ == "__main__":
    asyncio.run(main())