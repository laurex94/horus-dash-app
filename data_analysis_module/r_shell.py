import os

R_PATH = "C:\\Users\\elinarezv\\r.bat"
R_SCRIPT_IN = (
    "C:\\Users\\elinarezv\\Sources\\diogenet\\diogenet_py\\diogenet_py\\test.r"
)
R_SCRIPT_OUT = (
    "C:\\Users\\elinarezv\\Sources\\diogenet\\diogenet_py\\diogenet_py\\test.txt"
)

FULL_CMD = R_PATH + " <" + R_SCRIPT_IN + "> " + R_SCRIPT_OUT

os.system(FULL_CMD)

