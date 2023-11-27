with open("days/day01/input", "r") as f:
    data = f.read()

groups = []
counts = []
for line in data.splitlines():
    if line == "":
        groups.append(counts)
        counts = []
    else:
        counts.append(int(line))

totals = [sum(group) for group in groups]
totals.sort()

print(f"part 1: {totals[-1]}")
print(f"part 2: {sum(totals[-3:])}")
