import numpy as np
import franc.evaluation


# define test data

generator = franc.evaluation.NewtonianNoiseDataGenerator(folder="test")
#generator.generateEventSet(tag="train")
#generator.generateEventSet(tag="test")

#generator = franc.eval.TestDataGenerator([0.1]*n_channel, rng_seed=123)
witness_conditioning, target_conditioning= generator.generateDataset(tag="train")
witness_evaluation, target_evaluation = generator.generateDataset(tag="test")

print('witness shapes', [i.shape for i in witness_conditioning])
print('target shapes', [i.shape for i in target_conditioning])

sampling_rate = generator.default_tmax/generator.default_Nt

# create the dataset object
dataset = franc.evaluation.EvaluationDataset(
   sampling_rate,
   witness_conditioning,
   target_conditioning,
   witness_evaluation,
   target_evaluation
)