# -*- coding: utf-8 -*-
from typing import Dict, List

from ontomap.common.base import BaseDataset


class OMDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def is_contain_label(self, **kwargs) -> bool:
        pass

    def get_owl_items(self, **kwargs) -> List:
        pass

    def parse(self, **kwargs) -> List[Dict]:
        pass

    def get_name(self, **kwargs) -> str:
        pass

    def get_label(self, **kwargs) -> str:
        pass

    def get_iri(self, **kwargs) -> str:
        pass

    def collect_data(self, **kwargs) -> Dict:
        pass
