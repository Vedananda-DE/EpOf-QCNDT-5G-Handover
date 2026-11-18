function result = baseline_qcndt_goa(Xtrain, ytrain, Xval, yval, Xtest, params)
[theta, history] = qcndt_search(Xtrain, ytrain, Xval, yval, params, 'goa');
result.name = 'QCNDT-GOA';
result.model = theta;
result.labels = prediction(theta, Xtest);
result.history = history;
end
