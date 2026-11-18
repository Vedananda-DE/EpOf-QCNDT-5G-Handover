function result = baseline_qcndt_zoa(Xtrain, ytrain, Xval, yval, Xtest, params)
[theta, history] = qcndt_search(Xtrain, ytrain, Xval, yval, params, 'zoa');
result.name = 'QCNDT-ZOA';
result.model = theta;
result.labels = prediction(theta, Xtest);
result.history = history;
end
