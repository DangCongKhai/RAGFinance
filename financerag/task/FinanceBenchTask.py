from .BaseTask import BaseTask
from .TaskMetadata import TaskMetaData

class FinanceBenchTask(BaseTask):
    def __init__(self):
        metadata =  TaskMetaData(
            dataset_name = 'financebench',
            task_type = 'retrieval'
        )
        super().__init__(metadata)