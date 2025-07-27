# Example code to download the Pistols dataset from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="Z2wHDNyXcnGofRNc1Aln")
project = rf.workspace("joseph-nelson").project("pistols")
dataset = project.version(1).download("yolov8")
