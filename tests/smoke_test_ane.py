import warnings
warnings.filterwarnings('ignore')

from flexynesis.data import DataImporter
from flexynesis.main import HyperparameterTuning
from flexynesis.models.direct_pred import DirectPred
from flexynesis.utils import get_optimal_device

dev_str, dev_type = get_optimal_device('ane')
print(f"ANE device: str={repr(dev_str)}  type={repr(dev_type)}")

print("Loading dataset...")
importer = DataImporter(
    path="../flexynesis-ane/dataset1",
    data_types=["gex", "cnv"],
    covariates=["Crizotinib"],
    min_features=50,
)
dataset, test_dataset = importer.import_data()
print(f"Dataset loaded: {len(dataset.samples)} train samples, {len(test_dataset.samples)} test samples")

print("\nRunning DirectPred with device=ane (2 HPO iterations)...")
tuner = HyperparameterTuning(
    dataset=dataset,
    model_class=DirectPred,
    config_name="DirectPred",
    target_variables=["Crizotinib"],
    n_iter=2,
    device_type="ane",
)
best_model, best_params = tuner.perform_tuning()

from flexynesis.ane.linear import ane_compile_stats
stats = ane_compile_stats()
print(f"\nANE kernels compiled: {stats['kernels_compiled']}")
print(f"Unique shapes: {stats['shapes']}")
print(f"Model loss_scale: {best_model.loss_scale}")
print("\nDONE — flexynesis DirectPred trained on ANE")
