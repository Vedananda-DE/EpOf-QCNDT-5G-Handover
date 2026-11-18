function result = baseline_aho_drl(Xtrain, ytrain, Xtest, params)
result = actor_critic(Xtrain, ytrain, Xtest, params, 'aho');
end
