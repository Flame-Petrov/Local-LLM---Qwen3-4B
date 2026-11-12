import json
from pathlib import Path
p=Path(__file__).parent
r=json.loads((p/'results.json').read_text(encoding='utf-8'))
lines=['BLUETOOTH DISCOVERY REPORT',f"Scan UTC: {r['started_utc']} to {r['finished_utc']}",f"BLE addresses observed: {len(r['ble'])}",'Addresses are not necessarily unique physical devices. Classic entries may be remembered devices, not currently nearby.','', 'CLASSIC BLUETOOTH INQUIRY RESULTS', json.dumps(r['classic'],indent=2,ensure_ascii=False),'','LIVE BLUETOOTH LOW ENERGY ADVERTISEMENTS']
for d in sorted(r['ble'],key=lambda d:d['rssi_max'],reverse=True):
    lines.extend(['',f"{d['address']} | {', '.join(d['names']) or '(no name broadcast)'} | RSSI {d['rssi_min']} to {d['rssi_max']} dBm",json.dumps(d,indent=2,ensure_ascii=False)])
lines.extend(['','WINDOWS STORED DEVICE AND SERVICE RECORDS',(p/'windows-device-records.json').read_text(encoding='utf-8-sig')])
(p/'full-report.txt').write_text('\n'.join(lines),encoding='utf-8')
print('BLE count:',len(r['ble']))
print('Named:',json.dumps([{'address':d['address'],'names':d['names'],'rssi_max':d['rssi_max']} for d in r['ble'] if d['names']],ensure_ascii=True))
print('Classic:',json.dumps(r['classic'],ensure_ascii=True))
