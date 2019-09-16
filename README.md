# Problem Statement
The goal is to calculate the average face embedding vector across a large dataset of
faces. And make it extensible.

# Installation
There are multiple ways to install.
(You need build-essential and cmake installed on your system before installing requirements,.txt)

- Using setup.py
```
python setup.py test
python setup.py sdist
pip install dist/face_embedding-0.0.1-py3-none-any.whl
```
- Using makefile
```
make test
make buildproject
make install
```
- Using docker file
```
docker pull ishanbhatt/veriff:test
```

# Prerequisite
- Download the dataset which has similar structure to http://vis-www.cs.umass.edu/lfw/#download
- Extract it and put that in some location in your system.

# How to run
- Without docker
```
from face_embedding import *
embedded_vectors = get_all_cf("/path/to/downloaded_file")
average_vector = get_average(embedded_vectors)
```

- With docker (given that you have pulled the image)
```
docker run -v /local_path/to/downloaded_files:images ishanbhatt/veriff:test
```
You can also get individual celebrity face-vector as well for that you need to specify some settings using environment variable.
```
docker run -v /local_path/to/downloaded_files:/images -e METHOD=get_average -e CELEB=<celeb_name> ishanbhatt/veriff:test
```

# Sample output
```
[-0.09184992  0.08891898  0.05172213 -0.03960983 -0.09545068 -0.0170844
 -0.01534049 -0.10618819  0.13509281 -0.05730253  0.20093565 -0.03792449
 -0.2454025  -0.04531503 -0.02827421  0.12434064 -0.14732012 -0.11928621
 -0.12448212 -0.08079315  0.01167066  0.0335183   0.03045237  0.0167361
 -0.13223079 -0.2992694  -0.0701954  -0.08902419  0.05684252 -0.09198239
  0.01209295  0.05314682 -0.18470068 -0.0599186   0.02795471  0.06273955
 -0.05172922 -0.06855772  0.20853005  0.01110339 -0.16751085  0.01483111
  0.05365957  0.24574809  0.20452462  0.00596853  0.02080457 -0.07223468
  0.11354044 -0.26228943  0.04808586  0.144542    0.11229021  0.07523397
  0.0831255  -0.14238386  0.02096773  0.1370085  -0.18205424  0.06994481
  0.06932121 -0.09159455 -0.03761861 -0.04275434  0.16209593  0.0873592
 -0.09012995 -0.13499565  0.16634366 -0.1412926  -0.04414149  0.07672209
 -0.08983798 -0.15972088 -0.26815945  0.03592251  0.3752029   0.12747034
 -0.17584328  0.01636036 -0.07415752 -0.02344867  0.03634182  0.03259636
 -0.07445394 -0.04941602 -0.09306157  0.02423751  0.19745302 -0.02871042
 -0.01720607  0.21492124  0.010714   -0.00624987  0.04012334  0.04733949
 -0.11072838 -0.01356372 -0.10988516 -0.01219312  0.04726631 -0.10606343
  0.00958324  0.08684941 -0.17421512  0.15863739 -0.01775398 -0.02408041
 -0.00742908 -0.02445264 -0.07560085  0.00209789  0.17316285 -0.25108418
  0.21515241  0.17408757 -0.00231636  0.1254312   0.06280041  0.06165447
 -0.00458359 -0.00511429 -0.13308914 -0.10946756  0.02851166 -0.02758148
  0.02185735  0.03925822]
```

# Other
There are some assumptions and workarounds ,you can find them in NOTES.txt. 