import math
import logging
import torch
import torch.nn.functional as F


class DatasetEvaluator:
    """
    Class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device('cpu')

        self._predictions = []

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """

        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        targets = inputs["batched_targets"]
        preds = outputs["rgb"].to(self._cpu_device)
        prediction = {
            "targets": targets,
            "preds": preds,
        }
        self._predictions.append(prediction)

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            
        """
        if len(self._predictions) == 0:
            self._logger.warning("[DatasetEvaluator] Did not receive valid predictions.")
            return {}

        self._logger.info("Evaluating predictions")

        mse = 0.
        for prediction in self._predictions:
            targets = prediction["targets"]
            preds = prediction["preds"]

            mse += F.mse_loss(preds, targets, reduction='mean')
        
        mse = mse / len(self._predictions)
        psnr = -10. * math.log10(mse)

        results = {"MSE": mse, "PSNR": psnr}
        return results
