import os
print(os.path.expanduser("~"))
userhome = os.path.expanduser("~").split("/")[-1]
print(userhome)
