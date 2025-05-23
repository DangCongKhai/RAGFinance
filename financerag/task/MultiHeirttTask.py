from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData


class MultiHeirttTask(BaseTask):
    def __init__(self):
        metadata = TaskMetaData(
            dataset_name = 'multiheirtt',
            task_type = 'retrieval'
        )
        super().__init__(metadata)
