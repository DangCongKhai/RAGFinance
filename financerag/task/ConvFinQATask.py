from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData

class ConvFinQATask(BaseTask):
    def __init__(self):
        metadata = TaskMetaData(
            dataset_name = "convfinqa",
            task_type = 'retrieval')
        super().__init__(metadata)

