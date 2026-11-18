function result = baseline_ddpg(Xtrain, ytrain, Xtest, params)
result = actor_critic(Xtrain, ytrain, Xtest, params, 'ddpg');
end
