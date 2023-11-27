with open("days/day02/input", "r") as f:
    data = f.read()

decode_move = {
    "A": "rock",
    "B": "paper",
    "C": "scissors",
    "X": "rock",
    "Y": "paper",
    "Z": "scissors",
}
move_score = {
    "rock": 1,
    "paper": 2,
    "scissors": 3,
}
result_score = {
    ("rock", "paper"): 0,
    ("rock", "scissors"): 6,
    ("rock", "rock"): 3,
    ("paper", "scissors"): 0,
    ("paper", "rock"): 6,
    ("paper", "paper"): 3,
    ("scissors", "rock"): 0,
    ("scissors", "paper"): 6,
    ("scissors", "scissors"): 3,
}
decode_result = {
    "X": "lose",
    "Y": "draw",
    "Z": "win",
}
move_lookup = {
    ("rock", "X"): "scissors",
    ("rock", "Y"): "rock",
    ("rock", "Z"): "paper",
    ("paper", "X"): "rock",
    ("paper", "Y"): "paper",
    ("paper", "Z"): "scissors",
    ("scissors", "X"): "paper",
    ("scissors", "Y"): "scissors",
    ("scissors", "Z"): "rock",
}
score = 0
for line in data.splitlines():
    opp_move, move = line.split(" ")
    opp_move, move = decode_move[opp_move], decode_move[move]
    score += move_score[move]
    score += result_score[(move, opp_move)]
print(f"part1: {score}")

score = 0
for line in data.splitlines():
    opp_move, result = line.split(" ")
    opp_move = decode_move[opp_move]
    move = move_lookup[(opp_move, result)]
    score += move_score[move]
    score += result_score[(move, opp_move)]
print(f"part2: {score}")
