
### nsmc
negative=0
positive=0

input_file='./data/nsmc/ratings_test.txt'


with open(input_file, "r", encoding="utf-8") as f:
    lines = []
    for line in f:
        lines.append(line.strip())

for line in lines:
    line = line.split("\t")
    label = line[2]

    if label=="0":
        negative+=1
    elif label=='1':
        positive+=1

print(f'negative: {negative}')
print(f'postiive: {positive}')


'''
### korean hate speech
none=0
offensive=0
hate=0

input_file='./data/hate-speech/val.tsv'
with open(input_file, "r", encoding="utf-8") as f:
    lines = []
    for line in f:
        lines.append(line.strip())

for line in lines:
    line = line.split("\t")
    label = line[3]

    if label=="none":
        none+=1
    elif label=='offensive':
        offensive+=1
    elif label=='hate':
        hate+=1

print(f'none: {none}')
print(f'offensive: {offensive}')
print(f'hate: {hate}')
'''