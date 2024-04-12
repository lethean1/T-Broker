with open('time.txt', 'r') as f:
    lines = f.readlines()

data = []
buffer = []
tag = None
for line in lines:
    if line.startswith('['):
        if tag is not None:
            data.append((tag, buffer))
        tag = line
        buffer = []
    else:
        buffer.append(line) 

data.append((tag, buffer))

info_dict = {
    "[model]": [],
    "[time]": []
}

for d in data:
    info_dict[d[0].strip()].append(d[1])
new_time_info = []
for time_table in info_dict['[time]']:
    new_time_info.append([x for x in time_table if '[profile]' in x])
info_dict['[time]'] = new_time_info
with open('target_time.txt', 'w') as f:
    for m, t in zip(info_dict['[model]'], info_dict['[time]']):
        f.write('[model]\n')
        f.writelines(m)
        f.write('[time]\n')
        f.writelines(t)
    
    