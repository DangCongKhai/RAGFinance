from pydantic import BaseModel
from typing import List, Dict, Literal

TASK_TYPES = Literal['retrieval', 'generation']


class TaskMetaData(BaseModel):
    dataset_name : str
    task_type : TASK_TYPES

    
