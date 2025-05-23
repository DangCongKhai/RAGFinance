from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData


class FinQABenchTask(BaseTask):
    def __init__(self):
        metadata = TaskMetaData(
            dataset_name = 'finqabench',
            task_type = 'retrieval'
        )
        super().__init__(metadata)
