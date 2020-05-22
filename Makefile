help:
	@echo "venv      - Create tdlda_venv and install all necessary dependencies. Stays in venv afterwards."
#	@echo "venv_nemo - Create tdlda_venv and install all necessary dependencies on nemo."
#	@echo "test      - Run tests. Requires virtual env set up."

venv:
	rm -rf tdlda_venv ;\
	python3 -m venv tdlda_venv ;\
	. tdlda_venv/bin/activate ;\
	pip install --upgrade pip==19.3.1 ;\
	pip install -r requirements.txt ;\
	pip install -e .

#venv_nemo:
#	rm -rf tdlda_venv ;\
#	module load devel/python/3.6.9 ;\
#	python3 -m venv tdlda_venv --without-pip ;\
#	. tdlda_venv/bin/activate ;\
#	wget https://bootstrap.pypa.io/get-pip.py ;\
#	python get-pip.py ;\
#	rm get-pip.py ;\
#	pip install -r requirements.txt ;\
#	pip install -e .


#test:
#	. tdlda_venv/bin/activate ;\
#	python -m pytest tests/
