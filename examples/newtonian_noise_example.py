""" "An example to illustrate the dataset generation with the NewtonianNoiseDataGenerator"""

import franc.evaluation

# define test data
PATH = "test"  # no / at the end

generator = franc.evaluation.NewtonianNoiseDataGenerator(folder=PATH)
try:
    generator.generateEventSet(tag="train")
    generator.generateEventSet(tag="test")
except NameError:
    pass

witness_conditioning, target_conditioning = generator.generateDataset(tag="train")
witness_evaluation, target_evaluation = generator.generateDataset(tag="test")

print("witness shapes", [i.shape for i in witness_conditioning])
print("target shapes", [i.shape for i in target_conditioning])

sampling_rate = generator.default_tmax / generator.default_Nt

# create the dataset object
dataset = franc.evaluation.EvaluationDataset(
    sampling_rate,
    witness_conditioning,
    target_conditioning,
    witness_evaluation,
    target_evaluation,
)
