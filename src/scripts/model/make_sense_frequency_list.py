import tempfile
from pathlib import Path
from typing import Tuple, Any, Dict, Optional, List

import hydra
import torch
from omegaconf import omegaconf, DictConfig

from src.consec_dataset import ConsecSample
from src.dependency_finder import EmptyDependencyFinder
from src.pl_modules import ConsecPLModule
from src.scripts.model.continuous_predict import Predictor
from src.sense_inventories import SenseInventory, WordNetSenseInventory
from src.utils.commons import execute_bash_command
from src.utils.hydra import fix



def framework_evaluate(framework_folder: str, gold_file_path: str, pred_file_path: str) -> Tuple[float, float, float]:
    scorer_folder = f"{framework_folder}/Evaluation_Datasets"
    command_output = execute_bash_command(
        f"[ ! -e {scorer_folder}/Scorer.class ] && javac -d {scorer_folder} {scorer_folder}/Scorer.java; java -cp {scorer_folder} Scorer {gold_file_path} {pred_file_path}"
    )
    command_output = command_output.split("\n")
    p, r, f1 = [float(command_output[i].split("=")[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


def sample_prediction2sense(sample: ConsecSample, prediction: int, sense_inventory: SenseInventory) -> str:
    sample_senses = sense_inventory.get_possible_senses(
        sample.disambiguation_instance.lemma, sample.disambiguation_instance.pos
    )
    sample_definitions = [sense_inventory.get_definition(s) for s in sample_senses]

    for s, d in zip(sample_senses, sample_definitions):
        if d == sample.candidate_definitions[prediction].text:
            return s

    raise ValueError


def get_sense2frequency_map(
    amalgum_path: str,
    module: ConsecPLModule,
    predictor: Predictor,
    samples_generator: DictConfig,
    prediction_params: Dict[Any, Any],
    reporting_folder: Optional[str] = None,
) -> None:

    # load tokenizer
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    # instantiate samples
    consec_samples = list(hydra.utils.instantiate(samples_generator, dependency_finder=EmptyDependencyFinder())())

    # predict
    result = predictor.predict(
        consec_samples,
        already_kwown_predictions=None,
        reporting_folder=reporting_folder,
        **dict(module=module, tokenizer=tokenizer, **prediction_params),
    )
    with open(Path(amalgum_path).stem, "w") as f:
        for consec_sample, pred in result:
            f.write(
                f"{consec_sample.disambiguation_instance.text} | "
                f"{consec_sample.kwargs['instance_lemma']} | "
                f"{consec_sample.kwargs['instance_pos']}\n")
            f.write(f"{consec_sample.kwargs['instance_possible_definitions'][pred]}\n")
            f.write(f"{consec_sample.kwargs['instance_possible_senses'][pred]}\n\n")


@hydra.main(config_path="../../../conf/frequency_list", config_name="amalgum_frequency_list")
def main(conf: omegaconf.DictConfig) -> None:

    fix(conf)

    print(f"Generating the frequency list for {conf.amalgum_path}...")
    print()

    # load module
    module = ConsecPLModule.load_from_checkpoint(conf.model.model_checkpoint)
    module.to(torch.device(conf.model.device if conf.model.device != -1 else "cpu"))
    module.eval()
    module.freeze()
    module.sense_extractor.evaluation_mode = True  # no loss will be computed even if labels are passed

    # instantiate sense inventory
    sense_inventory = hydra.utils.instantiate(conf.sense_inventory)

    # instantiate predictor
    predictor = hydra.utils.instantiate(conf.predictor)

    # get frequency map
    get_sense2frequency_map(
        amalgum_path=conf.amalgum_path,
        module=module,
        predictor=predictor,
        samples_generator=conf.samples_generator,
        prediction_params=conf.model.prediction_params,
        reporting_folder="."
    )


if __name__ == "__main__":
    main()
