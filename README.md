# ganabi

Because DeepMind wrote their Rainbow agents in Py 2.7 and tf 1.x, the data creation script, which interfaces with that code, uses Py 2.7 and tf 1.x. However, once the data is produced, we only use Py 3.6 and tf 2.0 for building and training our models.

### Getting Started:
```
fork/clone repo into your home folder
cd ~/ganabi/hanabi-env
cmake .
make
```

### How to Run:
```
virtualenv venv2 --python==python2
virtualenv venv3 --python==python3

source /venv2/bin/activate

pip install -r requirements.txt

python create_data.py

source /venv3/bin/activate

pip install -r requirements.txt

python run_experiment.py -newrun --mode="naive_mlp" --configpath="./config.gin"
```