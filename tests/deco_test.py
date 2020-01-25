def initialize_deco(cls):
    cls.size = 20
    return cls

@initialize_deco
class Person:
    def __init__(self, name):
        self.name = name



p =Person('hello')
print(p.size)