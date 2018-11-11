# PRF

## installation:

clone the repository, and python setup.py install

## example usage (also see PRF/examples folder):  
from PRF import prf

prf_cls = prf(n_estimators=10,  bootstrap=True, keep_proba=0.05)

prf_cls.fit(X=X, dX=dX,y=y)

pred = prf_cls.predict(X=X, dX=dX)

https://nbviewer.jupyter.org/github/ireis/PRF/blob/master/PRF/examples/PRF_for_missing_data.ipynb
