import json, os

with open('models/metrics.json') as f:
    m = json.load(f)

print('=== LENS 1 ===')
for name, vals in m['lens1_revenue'].items():
    if name == 'champion_info':
        print('  CHAMPION:', vals)
        continue
    vw = vals['validation']['WAPE_percent']
    tw = vals['test']['WAPE_percent']
    print(f'  {name}: Val={vw}%  Test={tw}%')

print()
print('=== LENS 2 ===')
for name, vals in m['lens2_demand'].items():
    if name == 'champion_info':
        print('  CHAMPION:', vals)
        continue
    vw = vals['validation']['WAPE_percent']
    tw = vals['test']['WAPE_percent']
    print(f'  {name}: Val={vw}%  Test={tw}%')

print()
print('=== MODEL FILES ===')
for fname in sorted(os.listdir('models')):
    size_kb = os.path.getsize(os.path.join('models', fname)) // 1024
    print(f'  {fname}  ({size_kb} KB)')
