from typing import List, Tuple


class MeshObj:
    def __init__(self, label: str, plist: List[Tuple[int]], conn: 'Array', probtype: str):
        self.label = label
        self.plist = plist
        self.conn = conn
        self.probtype = probtype
