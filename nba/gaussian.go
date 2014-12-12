package nba

import (
	"errors"
	"math"
)

type (
	Point       []float64
	Class       string
	Probability float64

	GuassianDistribution struct {
		Mean  float64
		Sigma float64
	}

	Gaussian struct {
		dimensionality int
		classPriors    map[Class]Probability
		evidencePriors []GuassianDistribution
		classModel     map[Class][]GuassianDistribution
	}
)

const epsilon = 0.0001

var (
	WrongDimensionError   = errors.New("Dimensionality of data does not match prior data")
	NoDataError           = errors.New("Cannot fit model without training data")
	NoClassificationError = errors.New("No Class was found for this data point")

	sqrt2Pi = math.Sqrt(2.0 * math.Pi)
)

func (n *GuassianDistribution) Likelihood(x float64) float64 {
	return (1.0 / (n.Sigma * sqrt2Pi)) * math.Exp(-math.Pow(x-n.Mean, 2)/(2.0*math.Pow(n.Sigma, 2)))
}

func NewGaussian(dimensionality int) *Gaussian {
	return &Gaussian{
		dimensionality: dimensionality,
		classPriors:    make(map[Class]Probability),
		evidencePriors: make([]GuassianDistribution, dimensionality),
		classModel:     make(map[Class][]GuassianDistribution),
	}
}

func (nba *Gaussian) Fit(data map[Class][]Point) error {
	if len(data) < 1 {
		return NoDataError
	}

	totalPoints := 0
	var allPoints []Point
	for _, points := range data {
		if len(points[0]) != nba.dimensionality {
			return WrongDimensionError
		}
		totalPoints += len(points)
		allPoints = append(allPoints, points...)
	}

	for class, points := range data {
		// Calculate the prior of a class
		nba.classPriors[class] = Probability(float64(len(points)) / float64(totalPoints))

		// Calculate the mean and variance of each point dimension with respect its class
		mean := make(Point, nba.dimensionality)
		variance := make([]float64, nba.dimensionality)
		for i := 0; i < nba.dimensionality; i++ {
			// Calculate mean
			for _, p := range points {
				mean[i] += p[i]
			}
			mean[i] /= float64(len(points))

			// Calculate variance
			for _, p := range points {
				variance[i] += math.Pow(mean[i]-p[i], 2)
			}
			variance[i] /= float64(len(points))

			// Add epsilon to avoid 0 probability
			variance[i] += epsilon
		}

		// Make guassian distributions for class models
		nba.classModel[class] = make([]GuassianDistribution, nba.dimensionality)
		for i := 0; i < nba.dimensionality; i++ {
			nba.classModel[class][i] = GuassianDistribution{mean[i], math.Sqrt(variance[i])}
		}
	}

	// Calculate the total mean and variance of each point dimension
	mean := make(Point, nba.dimensionality)
	variance := make([]float64, nba.dimensionality)
	for i := 0; i < nba.dimensionality; i++ {
		// Calculate mean
		for _, p := range allPoints {
			mean[i] += p[i]
		}

		// Calculate variance
		mean[i] /= float64(len(allPoints))
		for _, p := range allPoints {
			variance[i] += math.Pow(mean[i]-p[i], 2)
		}
		variance[i] /= float64(len(allPoints))

		// Add epsilon to avoid 0 probability
		variance[i] += epsilon
	}

	// Make guassian distributions for evidence priors
	nba.evidencePriors = make([]GuassianDistribution, nba.dimensionality)
	for i := 0; i < nba.dimensionality; i++ {
		nba.evidencePriors[i] = GuassianDistribution{mean[i], math.Sqrt(variance[i])}
	}

	return nil
}

func (nba *Gaussian) Classify(point Point) (Class, error) {
	if len(point) != nba.dimensionality {
		return "", WrongDimensionError
	}

	// Get the total prior for the point
	var evidencePrior float64 = 1
	for i, prior := range nba.evidencePriors {
		evidencePrior *= prior.Likelihood(point[i])
	}

	var bestClass Class
	var bestClassProbability Probability
	for class, classPrior := range nba.classPriors {

		// Get the total class model for the point conditiond on this class
		var conditional float64 = 1
		for i, prior := range nba.classModel[class] {
			conditional *= prior.Likelihood(point[i])
		}

		// Bayes theorem: P(c|e) = (P(e|c)P(c)) / P(e)
		probability := Probability((conditional * float64(classPrior)) / evidencePrior)

		// Update current best class
		if probability > bestClassProbability {
			bestClassProbability = probability
			bestClass = class
		}
	}

	if bestClassProbability == 0 {
		return "", NoClassificationError
	}

	return bestClass, nil
}
