from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData


class FinDERTask(BaseTask):
    def __init__(self):
        metadata = TaskMetaData(
            dataset_name = 'finder',
            task_type = 'retrieval'
        )
        super().__init__(metadata)
