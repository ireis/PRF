# Probabilistic Random Forest (PRF)

The PRF is a modification the long-established Random Forest (RF) algorithm that takes into account uncertainties in the measurements (i.e., features) as well as in the assigned classes (i.e., labels). To do so, the Probabilistic Random Forest (PRF) algorithm treats the features and labels as probability distribution functions, rather than deterministic quantities. The details of the algorithm along with comparison to the original RF are described in the paper:

[Probabilistic Random Forest: A machine learning algorithm for noisy datasets](https://arxiv.org/abs/1811.05994v1)


## installation:

clone the repository and from PRF\ run
```
python setup.py install
```

Tested only on python 3, please email ```itamarreis@mail.tau.ac.il``` if you encounter any issues with installation or running the code.

## example usage   
```
from PRF import prf
prf_cls = prf(n_estimators=10, bootstrap=True, keep_proba=0.05)
prf_cls.fit(X=X_train, dX=dX_train, y=y_train)
pred = prf_cls.predict(X=X_test, dX=dX_test)
```

also see PRF/examples folder:

https://nbviewer.jupyter.org/github/ireis/PRF/blob/master/PRF/examples/PRF_for_missing_data.ipynb

## Authors

* **Itamar Reis** - https://github.com/ireis

* **Dalya Baron** - https://github.com/dalya

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
