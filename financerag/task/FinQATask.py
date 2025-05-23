from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData


class FinQATask(BaseTask):
    def __init__(self):
        metadata = TaskMetaData(
            dataset_name = 'finqa',
            task_type = 'retrieval'
        )
        super().__init__(metadata)
