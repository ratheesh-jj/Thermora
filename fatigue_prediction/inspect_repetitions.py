from pathlib import Path
import re
from collections import defaultdict

BASE = Path(
    r"C:\Users\OMEN\Desktop\Projects\SIH26\fatigue_prediction"
)

DATA = (
    BASE
    / "fatigue_personalization_results"
    / "extracted"
    / "MxD_Data_User_Study"
    / "MxD_Data_User_Study"
)

groups = defaultdict(list)

for f in DATA.rglob("Rep[1-5]_*.csv"):

    m = re.search(
        r"(P\d+C\d+S\d+)",
        f.name
    )

    if not m:
        continue

    recording = m.group(1)

    # P001C001S002 -> P001
    participant = recording.split("C")[0]

    # composite / ziptie
    task = f.parent.parent.name

    rep = re.search(
        r"Rep(\d+)_",
        f.name
    )

    if rep:
        groups[(task, participant)].append(
            int(rep.group(1))
        )

print("=" * 70)
print("MxD REPETITION STRUCTURE")
print("=" * 70)

total = 0

for (task, participant), reps in sorted(groups.items()):

    reps = sorted(set(reps))

    print(
        f"{task:10s} {participant:5s} "
        f"{len(reps)} reps -> {reps}"
    )

    total += len(reps)

print()
print("TOTAL REPETITION FILES:", total)

print()
print("=" * 70)
print("GROUP COUNTS")
print("=" * 70)

for task in sorted(set(k[0] for k in groups)):

    participants = [
        k for k in groups
        if k[0] == task
    ]

    reps = sum(
        len(groups[k])
        for k in participants
    )

    print(
        task,
        "participants =", len(participants),
        "repetitions =", reps
    )

print()
print("=" * 70)
print("39-PARTICIPANT / 195-SAMPLE CANDIDATES")
print("=" * 70)

for task in sorted(set(k[0] for k in groups)):

    candidates = []

    for (t, participant), reps in groups.items():

        if t == task and len(set(reps)) == 5:
            candidates.append(participant)

    print(
        task,
        "participants with exactly 5 reps:",
        len(candidates)
    )

    print(
        sorted(candidates)
    )