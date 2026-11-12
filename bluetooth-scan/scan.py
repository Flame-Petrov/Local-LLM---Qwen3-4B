import asyncio, ctypes as c, json, sys
from ctypes import wintypes as w
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '.bluetooth-scan-deps'))

class ST(c.Structure):
    _fields_ = [(n,w.WORD) for n in ('year','month','weekday','day','hour','minute','second','milliseconds')]
class Info(c.Structure):
    _fields_ = [('size',w.DWORD),('address',c.c_ulonglong),('device_class',w.ULONG),('connected',w.BOOL),('remembered',w.BOOL),('authenticated',w.BOOL),('last_seen',ST),('last_used',ST),('name',w.WCHAR*248)]
class Search(c.Structure):
    _fields_ = [('size',w.DWORD),('authenticated',w.BOOL),('remembered',w.BOOL),('unknown',w.BOOL),('connected',w.BOOL),('inquiry',w.BOOL),('timeout',c.c_ubyte),('radio',w.HANDLE)]
def classic():
    dll=c.WinDLL('bthprops.cpl',use_last_error=True)
    dll.BluetoothFindFirstDevice.argtypes=[c.POINTER(Search),c.POINTER(Info)]
    dll.BluetoothFindFirstDevice.restype=w.HANDLE
    dll.BluetoothFindNextDevice.argtypes=[w.HANDLE,c.POINTER(Info)]
    dll.BluetoothFindDeviceClose.argtypes=[w.HANDLE]
    results=[]
    for attempt in range(2):
        s=Search(c.sizeof(Search),True,True,True,True,True,12,None)
        i=Info(); i.size=c.sizeof(Info)
        h=dll.BluetoothFindFirstDevice(c.byref(s),c.byref(i))
        if not h:
            results.append({'error':c.get_last_error(),'round':attempt+1}); continue
        try:
            while True:
                row={k:getattr(i,k) for k in ('name','device_class','connected','remembered','authenticated')}
                row['address']=':'.join(f'{i.address:012X}'[j:j+2] for j in range(0,12,2))
                for k in ('last_seen','last_used'):
                    t=getattr(i,k); row[k]={n:getattr(t,n) for n,_ in ST._fields_}
                row['round']=attempt+1
                results.append(row)
                if not dll.BluetoothFindNextDevice(h,c.byref(i)): break
        finally: dll.BluetoothFindDeviceClose(h)
    return results

async def main():
    from bleak import BleakScanner
    started=datetime.now(timezone.utc).isoformat()
    devices={}
    def receive(d,a):
        now=datetime.now(timezone.utc).isoformat()
        r=devices.setdefault(d.address,{'address':d.address,'names':[],'first_seen':now,'packets':0,'rssi_min':a.rssi,'rssi_max':a.rssi,'advertisements':[]})
        r['last_seen']=now; r['packets']+=1
        r['rssi_min']=min(r['rssi_min'],a.rssi); r['rssi_max']=max(r['rssi_max'],a.rssi)
        for name in (d.name,a.local_name):
            if name and name not in r['names']: r['names'].append(name)
        data={'local_name':a.local_name,'tx_power':a.tx_power,'service_uuids':a.service_uuids,'manufacturer_data':{str(k):v.hex() for k,v in a.manufacturer_data.items()},'service_data':{k:v.hex() for k,v in a.service_data.items()}}
        if data not in r['advertisements']: r['advertisements'].append(data)
    task=asyncio.create_task(asyncio.to_thread(classic))
    async with BleakScanner(receive,scanning_mode='active'):
        for n in range(6):
            await asyncio.sleep(15)
            print(f'{(n+1)*15}s: {len(devices)} BLE addresses detected',flush=True)
    result={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'ble':list(devices.values()),'classic':await task}
    path=Path(__file__).with_name('results.json')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)
asyncio.run(main())
