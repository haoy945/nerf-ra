import logging
import torch

from .evaluation import DatasetEvaluator

__all__ = ["inference", ]


@torch.no_grad()
def inference(model, data_loader):
    logger = logging.getLogger(__name__)
    total = len(data_loader)
    logger.info("Start inference on {} images".format(total))

    training_mode = model.training
    model.eval()
    evaluator = DatasetEvaluator()

    for idx, inputs in enumerate(data_loader):
        outputs = model(inputs)

        evaluator.process(inputs, outputs)

        if (idx + 1) % 1 == 0:
            logger.info("Inference done {}/{}. ".format(idx + 1, total))


    results = evaluator.evaluate()
    model.train(training_mode)
    return results
