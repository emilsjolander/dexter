package nba

import "math"

const epsilon = 0.0001

type (
	Point []float64

	GuassianDistribution struct {
		Mean  float64
		Sigma float64
	}

	Gaussian struct {
		dimensionality int
		classPriors    map[string]float64
		classModel     map[string][]GuassianDistribution
	}
)

var sqrt2Pi = math.Sqrt(2.0 * math.Pi)

func (n *GuassianDistribution) Likelihood(x float64) float64 {
	return (1.0 / (n.Sigma * sqrt2Pi)) * math.Exp(-(x-n.Mean)*(x-n.Mean)/(2.0*n.Sigma*n.Sigma))
}

func NewGaussian(dimensionality int) *Gaussian {
	return &Gaussian{
		dimensionality: dimensionality,
		classPriors:    make(map[string]float64),
		classModel:     make(map[string][]GuassianDistribution),
	}
}

func (nba *Gaussian) Fit(data map[string][]Point) error {
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
		nba.classPriors[class] = float64(len(points)) / float64(totalPoints)

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
				variance[i] += (mean[i] - p[i]) * (mean[i] - p[i])
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

	return nil
}

func (nba *Gaussian) Classify(point Point) (string, error) {
	if len(point) != nba.dimensionality {
		return "", WrongDimensionError
	}

	var bestClass string
	var bestClassLogProbability float64 = -math.MaxFloat64

	for class, classPrior := range nba.classPriors {

		// Get the total class model for the point conditiond on this class
		var logSum float64 = 0
		for i, prior := range nba.classModel[class] {
			logSum += math.Log(prior.Likelihood(point[i]))
		}

		// Bayes theorem: P(c|e) = (P(e|c)P(c)) / P(e)
		// We drop P(e) as it is constant
		logProbability := logSum + math.Log(classPrior)

		// Update current best class
		if logProbability > bestClassLogProbability {
			bestClassLogProbability = logProbability
			bestClass = class
		}
	}

	if bestClassLogProbability == -math.MaxFloat64 {
		return "", NoClassificationError
	}

	return bestClass, nil
}
