package ann

import (
	"errors"
	"math"
	"math/rand"
)

type ActivationFunction interface {
	Calc(float64) float64
	CalcDerivative(float64) float64
}

type SigmoidActivation struct{}

func (a *SigmoidActivation) Calc(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (a *SigmoidActivation) CalcDerivative(x float64) float64 {
	return a.Calc(x) * (1 - a.Calc(x))
}

type ReLActivation struct{}

func (a *ReLActivation) Calc(x float64) float64 {
	return math.Max(0, x)
}

func (a *ReLActivation) CalcDerivative(x float64) float64 {
	return math.Max(0, x) / x
}

type TanhActivation struct{}

func (a *TanhActivation) Calc(x float64) float64 {
	return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
}

func (a *TanhActivation) CalcDerivative(x float64) float64 {
	return 1 - math.Pow(a.Calc(x), 2)
}

type FeedForward struct {
	LearningRate float64
	layers       []Layer
}

type Layer []*Neuron

type Neuron struct {
	activationFunc ActivationFunction
	activation     float64
	signal         float64
	delta          float64
	in             []*edge
	out            []*edge
}

type edge struct {
	weight float64
	in     *Neuron
	out    *Neuron
}

var InputDimensionMismatchError = errors.New("Dimension of input must match the dimension on the input layer")
var OutputDimensionMismatchError = errors.New("Dimension of output must match the dimension on the output layer")
var NotInitializedError = errors.New("At least 2 layers must be added to the nextwork before training of predicting")

func NewFeedForward(learningRate float64, layers ...Layer) FeedForward {
	net := FeedForward{LearningRate: learningRate, layers: layers}

	// Connect layers
	for i := 0; i < len(net.layers)-1; i++ {
		this := net.layers[i]
		next := net.layers[i+1]

		// Initialize edge slices
		for _, n := range this {
			n.out = make([]*edge, 0, len(next))
		}
		for _, n := range next {
			n.in = make([]*edge, 0, len(this))
		}

		// Connect layers and initialize weights
		for _, thisNode := range this {
			for _, nextNode := range next {
				edge := &edge{in: thisNode, out: nextNode}
				for edge.weight == 0 {
					edge.weight = rand.NormFloat64()
				}
				edge.in.out = append(edge.in.out, edge)
				edge.out.in = append(edge.out.in, edge)
			}
		}
	}

	// Initialize and add bias units
	for i := 0; i < len(net.layers)-1; i++ {
		next := net.layers[i+1]

		bias := &Neuron{activationFunc: nil, signal: 1, activation: 1}
		bias.out = make([]*edge, 0, len(next))

		for _, nextNode := range next {
			edge := &edge{in: bias, out: nextNode}
			for edge.weight == 0 {
				edge.weight = rand.Float64()
			}
			edge.in.out = append(edge.in.out, edge)
			edge.out.in = append(edge.out.in, edge)
		}

		net.layers[i] = append(net.layers[i], bias)
	}

	return net
}

func NewLayer(numNeurons int, activationFunc ActivationFunction) Layer {
	layer := Layer(make([]*Neuron, numNeurons))
	for i := 0; i < len(layer); i++ {
		layer[i] = &Neuron{activationFunc: activationFunc}
	}
	return layer
}

func NewInputLayer(numNeurons int) Layer {
	return NewLayer(numNeurons, nil)
}

func (net *FeedForward) Train(in []float64, out []float64) ([]float64, error) {
	prediction, err := net.Predict(in)
	if err != nil {
		return []float64{}, err
	}
	if len(out) != len(prediction) {
		return []float64{}, OutputDimensionMismatchError
	}

	// Calculate gradients for output nodes
	outputLayer := net.layers[len(net.layers)-1]
	for i, n := range outputLayer {
		// Derivative of ((f(x) - y)^2) / 2
		n.delta = (prediction[i] - out[i]) * n.activationFunc.CalcDerivative(n.signal)
	}

	// Calculate gradients for rest of nodes
	for i := len(net.layers) - 2; i > 0; i-- {
		for _, n := range net.layers[i] {
			if n.activationFunc != nil {
				var sum float64
				for _, e := range n.out {
					sum += e.out.delta * e.weight
				}
				n.delta = n.activationFunc.CalcDerivative(n.signal) * sum
			}
		}
	}

	// Adjust weights along the negative of the gradient
	for i := len(net.layers) - 2; i >= 0; i-- {
		for _, n := range net.layers[i] {
			for _, e := range n.out {
				e.weight -= net.LearningRate * e.in.activation * e.out.delta
			}
		}
	}

	// Return a square error to the caller so that they can choose when to stop training
	sqError := make([]float64, len(out))
	for i := 0; i < len(out); i++ {
		sqError[i] = math.Pow(prediction[i]-out[i], 2)
	}
	return sqError, nil
}

func (net *FeedForward) Predict(in []float64) ([]float64, error) {
	// Make sure network is initialized
	if len(net.layers) < 2 {
		return []float64{}, NotInitializedError
	}

	// -1 adjusts for bias node
	if len(in) != len(net.layers[0])-1 {
		return []float64{}, InputDimensionMismatchError
	}

	// Activate the first layer
	for i := 0; i < len(in); i++ {
		n := net.layers[0][i]
		n.signal = in[i]
		n.activation = n.signal
	}

	// Feed forward
	for i := 1; i < len(net.layers); i++ {
		for _, n := range net.layers[i] {
			if n.activationFunc != nil {
				var sum float64
				for _, e := range n.in {
					sum += e.in.activation * e.weight
				}
				n.signal = sum
				n.activation = n.activationFunc.Calc(n.signal)
			}
		}
	}

	// Grab output
	outputLayer := net.layers[len(net.layers)-1]
	out := make([]float64, len(outputLayer))
	for i, n := range outputLayer {
		out[i] = n.activation
	}

	return out, nil
}
