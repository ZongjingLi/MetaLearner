

from datasets.numbers_dataset import get_dataset

dataset = get_dataset()

from core.learn import optimal_schedule


words, schedule = optimal_schedule(dataset, [])

print(words)

print(schedule)

