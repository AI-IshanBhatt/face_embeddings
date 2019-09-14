builddep:
	apt-get update && apt-get install -y build-essential cmake
	pip install -r requirements.txt


buildproject: builddep
	python setup.py sdist bdist_wheel

test: builddep 
	python setup.py test

install:
	pip install dist/face_embedding-0.0.1-py3-none-any.whl

