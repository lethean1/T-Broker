import sys
input_list = []
try:
    tmp = input()
    input_list += tmp.split(' ')
except EOFError:
    pass
begin_time_list = input_list[:len(input_list)//2]
end_time_list = input_list[len(input_list)//2:]
#with open('begin.txt') as f:
#    begin_time_list = f.readlines()
begin_time_list = [float(x) for x in begin_time_list]
#with open('end.txt') as f:
#    end_time_list = f.readlines()
end_time_list = [float(x) for x in end_time_list]

time0 = begin_time_list[0]
if len(sys.argv) == 2:
    base = int(sys.argv[1])
else:
    base = 0
#print(base)
qt_average = (sum([x-time0 for x in begin_time_list])-base) /len(begin_time_list)
jct_average = (sum([x-time0 for x in end_time_list])-base) /len(begin_time_list)
print(f"qt: {qt_average}")
print(f"jct: {jct_average}")
print(f"makespan: {end_time_list[-1] - begin_time_list[0]}")
#print(f"{qt_average}")
#print(f"{jct_average}")
#print(f"{end_time_list[-1] - begin_time_list[0]}")
