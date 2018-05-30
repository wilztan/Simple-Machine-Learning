from numpy import exp,array,random,dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1);
        self.synaptic_weight = 2 * random.random((3,1)) - 1;

    def __sigmoid(self,x):
        return 1/(1+exp(-x));

    def __sigmoid_derivative(self,x):
        return x * (1-x)

    def train(self,training_set_inputs,training_set_output,itterations):
        for itterations in range(0,itterations):
            output = self.predict(training_set_inputs);
            error = training_set_output - output;
            adjustment = dot(training_set_inputs.T,error * self.__sigmoid_derivative(output));
            self.synaptic_weight+=adjustment;

    def predict(self,input):
        return self.__sigmoid((dot(input,self.synaptic_weight)))


if __name__ == '__main__':

    neural_network = NeuralNetwork();

    print ('Random Starting Synaptic Weight');
    print (neural_network.synaptic_weight);

    training_set_inputs = array([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [1,0,0],
        [0,1,1],
        [1,1,0],
        [1,0,1],
        [1,1,1]
    ]);

    training_set_output = array([[
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1
    ]]).T;

    neural_network.train(training_set_inputs,training_set_output,10000);

    print ('New sypnatic weight after training :');
    print( neural_network.synaptic_weight);


    print ('Predicting');
    result = []
    a = float(input('number 1 : [0:1]'))
    b = float(input('number 2 : [0:1]'))
    c = float(input('number 3 : [0:1]'))
    result.append(a);
    result.append(b);
    result.append(c);
    print( neural_network.predict(array(result)));
