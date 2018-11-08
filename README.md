# PRF

## installation

clone the repository, and
python setup.py install

example usage:  
from PRF import prf
prf_cls = prf(n_estimators=n_trees,  bootstrap=True, keep_proba=0.05)
prf_cls.fit(X=X, dX=dX,y=y)
pred = prf_cls.predict(X=X, dX=dX)
