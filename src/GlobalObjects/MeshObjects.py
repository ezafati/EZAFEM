from typing import List


class MeshObj:
    def __init__(self, plist: List['Point'], conn: 'Array', probtype: str):
        self.plist = plist
        self.conn = conn
        self.probtype = probtype
