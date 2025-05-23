from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData


class TATQATask(BaseTask):
    def __init__(self):
        metadata = TaskMetaData(
            dataset_name = 'tatqa',
            task_type = 'retrieval'
        )
        super().__init__(metadata)
