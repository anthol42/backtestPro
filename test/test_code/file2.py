"""
This is a file deisgned to test code extarction.  It is not meant to be run.
"""
from enum import Enum

class TestObj(Enum):
    A = 'A'
    B = 'B'
    C = 'C'

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str):
        if string == 'A':
            return cls.A
        elif string == 'B':
            return cls.B
        elif string == 'C':
            return cls.C
        else:
            raise ValueError(f"Invalid string: {string}")


print("TestObj.A: ", TestObj.A)
print("TestObj.A.value: ", TestObj.A.value)
print("TestObj.A.name: ", TestObj.A.name)
print("TestObj.A.__str__(): ", TestObj.A.__str__())
print("TestObj.from_str('A'): ", TestObj.from_str('A'))

# ----------------------------------------------

print("TestObj.B: ", TestObj.B)
print("TestObj.B.value: ", TestObj.B.value)
print("TestObj.B.name: ", TestObj.B.name)
print("TestObj.B.__str__(): ", TestObj.B.__str__())
print("TestObj.from_str('B'): ", TestObj.from_str('B'))

# ----------------------------------------------

print("TestObj.C: ", TestObj.C)
print("TestObj.C.value: ", TestObj.C.value)
print("TestObj.C.name: ", TestObj.C.name)
print("TestObj.C.__str__(): ", TestObj.C.__str__())
print("TestObj.from_str('C'): ", TestObj.from_str('C'))

# ----------------------------------------------

try:
    TestObj.from_str('D')
except Exception as e:
    print(f"Unable to load type from 'D' (Which is normal)")
    print(e)