from matplotlib import pyplot as plt

template = 'log/elbo_%d/log.txt'
sizes = [1000, 5000, 10000, 50000]
values = {}

for size in sizes:
    reader = open(template % size, 'r')
    while True:
        line = reader.readline().split()
        if len(line) < 3:
            break
        if str(line[0]) not in values:
            values[str(line[0])] = [[size], [float(line[1])]]
        else:
            values[str(line[0])][0].append(size)
            values[str(line[0])][1].append(float(line[1]))


