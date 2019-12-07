from unityagents import UnityEnvironment
import keras
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import random
import per

def sampleEnvironment(targetModel,predictionModel,env,env_info,brain_name, epsilon, replayMemory,discountFactor):
    #(state,action,reward,next state)
    state = env_info.vector_observations[0]
    done = False
    samples = []
    score = 0
    while not done:
        state_batch = np.expand_dims(state, axis=0)
        predictionResult = predictionModel.predict(state_batch).flatten()
        bestAction = int(np.argmax(predictionResult))
        randomFloat = random.random()
        if randomFloat > epsilon:
            actionToTake = bestAction
        else:
            actionToTake = np.random.randint(action_size)
        env_info = env.step(actionToTake)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        score += reward
        done = env_info.local_done[0]  # see if episode has finished
        samples.append((state,actionToTake,reward,next_state,done))
        state = next_state
    states,values,errors = computeTargetValues(samples,targetModel,predictionModel,discountFactor)
    #add the samples to the prioritized replay memory
    for currentSampleIndex in range(len(samples)):
        replayMemory.add(errors[currentSampleIndex],samples[currentSampleIndex])
        #replayMemory.append(samples[currentSampleIndex])
    return score


def computeTargetValues(samples,targetModel,predictionModel,discountFactor):
    states = []
    targetValues = []
    errors = []
    for sample in samples:
        currentState = sample[0]
        action = sample[1]
        reward = sample[2]
        nextState = sample[3]
        done = sample[4]
        predictions_currentState = predictionModel.predict(np.expand_dims(currentState,axis=0)).flatten()
        predictions_nextState = targetModel.predict(np.expand_dims(nextState,axis=0)).flatten()
        maxReward = np.max(predictions_nextState)
        targetValue = predictions_currentState

        correctValue = reward + discountFactor*maxReward*(1-done)
        error = np.square(targetValue[action]- correctValue)
        errors.append(error)
        targetValue[action] = correctValue
        targetValues.append(targetValue)
        states.append(currentState)
    return np.array(states),np.array(targetValues),np.array(errors)

env = UnityEnvironment(file_name="Banana.exe")
#env = UnityEnvironment(file_name="...")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


a = Input(shape=(state_size,))
b = Dense(64,activation='relu')(a)
c = Dense(32,activation='relu')(b)
d = Dense(action_size,activation='linear')(c)

targetModel = Model(inputs=a, outputs=d)
predictionModel = keras.models.clone_model(targetModel)
predictionModel.set_weights(targetModel.get_weights())
predictionModel.summary()
predictionModel.compile(loss='mse', optimizer='adam', metrics=['mae'])

nEpisodes = 10000
stepsBeforeUpdate = 5
epsilon = 1
decay_factor = 0.999
samplePeriod = 10
batchSize = 64
sampleInterval = 100
discountFactor = 0.99
weightUpdateInterval = 10
minEpsilon = 0.01

replayMemory = per.Memory(100000)
#replayMemory = []

trainMode = True

if trainMode:

    accumScore = 0
    for nEpisodes in range(nEpisodes):
        env_info = env.reset(train_mode=True)[brain_name]
        epsilon *= decay_factor
        epsilon = max(epsilon, minEpsilon)
        print(epsilon)
        score = sampleEnvironment(targetModel, predictionModel, env, env_info, brain_name, epsilon, replayMemory, discountFactor)
        accumScore += score
        if nEpisodes > 0 and nEpisodes%100 == 0:
            averageScore = accumScore/100
            print("Average score for epoch {0}:{1}".format(nEpisodes,averageScore))
            if averageScore > 13.0:
                print("Environment solved in {0} episodes!".format(nEpisodes))
                predictionModel.save("models/bestModel.model")
                break
            accumScore = 0

        samples, idxs, is_weight = replayMemory.sample(batchSize)
        #samples = random.sample(replayMemory,batchSize)
        states, targetValues, errors = computeTargetValues(samples,targetModel,predictionModel,discountFactor)
        #get predictions first to
        predictionModel.fit(states,targetValues)
        newTargetValues = predictionModel.predict(states)
        newErrors = np.square(newTargetValues-targetValues)


        for newErrorIndex in range(len(newErrors)):
            replayMemory.update(idxs[newErrorIndex],newErrors[newErrorIndex][samples[newErrorIndex][1]])


        if nEpisodes > 0 and nEpisodes % weightUpdateInterval == 0:
            print("Copying weights...")
            targetModel.set_weights(predictionModel.get_weights())
            print("Target model weights:\n")

else:
    predictionModel = keras.models.load_model("models/bestKerasModel.model")
#simulate agent with learnt policy

    env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]
    score = 0  # initialize the score
    while True:
        state_batch = np.expand_dims(state, axis=0)
        predictedActionValues = predictionModel.predict(state_batch).flatten()
        bestAction = int(np.argmax(predictedActionValues))
        print("best action: {0}".format(bestAction))
        env_info = env.step(bestAction)[brain_name]  # send the action to the environment
        state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        print("Reward: {0}".format(reward))
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

    env.close()